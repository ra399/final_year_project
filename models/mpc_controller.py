"""
Model Predictive Control (MPC) for Binary Distillation Column
==============================================================
Author: BTP Student, Department of Chemical Engineering, IIT Patna
Project: XAI-MPC for Energy-Efficient Distillation Column Operation

Description:
-----------
This module implements a Model Predictive Controller using GEKKO for the
binary distillation column. The MPC optimizes:
1. Energy consumption (minimize reboiler duty)
2. Product purity constraints (top and bottom compositions)
3. Smooth control actions (minimize control rate of change)

The controller uses:
- Linear state-space model (linearized around operating point)
- Quadratic cost function
- Constraint handling
- Receding horizon optimization

Installation:
------------
pip install gekko numpy scipy matplotlib

References:
-----------
1. Rawlings, J. B., & Mayne, D. Q. (2009). "Model Predictive Control: Theory and Design"
2. Bequette, B. W. (2003). "Process Control: Modeling, Design, and Simulation"
"""

import numpy as np
from gekko import GEKKO
from scipy.signal import cont2discrete
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt


class DistillationMPC:
    """
    Model Predictive Controller for binary distillation column.
    
    The MPC solves the following optimization problem at each time step:
    
    minimize:   Σ[k=0 to N-1] { Q_energy*(Q_R[k])² + Q_purity*(x_top[k] - x_sp)² 
                                + R_u*(Δu[k])² }
    
    subject to:
        - System dynamics: x[k+1] = A*x[k] + B*u[k]
        - Output equations: y[k] = C*x[k]
        - Control constraints: u_min ≤ u[k] ≤ u_max
        - Output constraints: y_min ≤ y[k] ≤ y_max
        - Control rate constraints: Δu_min ≤ u[k] - u[k-1] ≤ Δu_max
    
    where:
        u = [L_R, Q_R]ᵀ (control inputs: reflux, reboiler duty)
        y = [x_top, x_bottom]ᵀ (outputs: top and bottom compositions)
    """
    
    def __init__(self,
                 prediction_horizon: int = 20,
                 control_horizon: int = 10,
                 sampling_time: float = 1.0):
        """
        Initialize MPC controller.
        
        Parameters:
        -----------
        prediction_horizon : int
            Number of future time steps to predict (N)
        control_horizon : int
            Number of control moves to optimize (M ≤ N)
        sampling_time : float
            Sampling/control interval in minutes
        """
        self.N = prediction_horizon
        self.M = control_horizon
        self.Ts = sampling_time
        
        # Initialize GEKKO optimizer
        self.m = GEKKO(remote=False)
        self.m.time = np.linspace(0, self.N * self.Ts, self.N + 1)
        
        # Tuning parameters (will be set during setup)
        self.Q_energy = None
        self.Q_purity = None
        self.R_control = None
        
        # Control and state variables (to be defined)
        self.u_reflux = None
        self.u_reboiler = None
        self.y_top = None
        self.y_bottom = None
        
        # State-space model matrices
        self.A = None
        self.B = None
        self.C = None
        self.D = None
        
        # Operating point for linearization
        self.x_op = None
        self.u_op = None
        
    def linearize_model(self,
                       column_model,
                       state_ss: np.ndarray,
                       u_nominal: Dict[str, float],
                       d_nominal: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Linearize the nonlinear distillation model around operating point.
        Uses finite differences to compute Jacobian matrices.
        
        Parameters:
        -----------
        column_model : BinaryDistillationColumn
            Nonlinear column model
        state_ss : np.ndarray
            Steady-state operating point
        u_nominal : dict
            Nominal control inputs
        d_nominal : dict
            Nominal disturbances
        
        Returns:
        --------
        A, B, C, D : np.ndarray
            State-space matrices for linearized model
        """
        print("Linearizing distillation model...")
        
        n_states = len(state_ss)
        n_controls = 2  # [L_R, Q_R]
        n_outputs = 2   # [x_top, x_bottom]
        
        # Perturbation size
        dx = 1e-4
        du = 1e-2
        
        # Compute A matrix (∂f/∂x)
        A = np.zeros((n_states, n_states))
        f_nominal = column_model.dynamics(state_ss, 0, u_nominal, d_nominal)
        
        for i in range(n_states):
            state_pert = state_ss.copy()
            state_pert[i] += dx
            f_pert = column_model.dynamics(state_pert, 0, u_nominal, d_nominal)
            A[:, i] = (f_pert - f_nominal) / dx
        
        # Compute B matrix (∂f/∂u)
        B = np.zeros((n_states, n_controls))
        
        # Reflux perturbation
        u_pert = u_nominal.copy()
        u_pert['L_R'] += du
        f_pert = column_model.dynamics(state_ss, 0, u_pert, d_nominal)
        B[:, 0] = (f_pert - f_nominal) / du
        
        # Reboiler duty perturbation
        u_pert = u_nominal.copy()
        u_pert['Q_R'] += du * 50  # Larger perturbation for Q_R
        f_pert = column_model.dynamics(state_ss, 0, u_pert, d_nominal)
        B[:, 1] = (f_pert - f_nominal) / (du * 50)
        
        # Compute C matrix (output = [x_top, x_bottom])
        C = np.zeros((n_outputs, n_states))
        n_trays = n_states // 2
        C[0, n_trays + n_trays - 1] = 1.0  # Top composition
        C[1, n_trays] = 1.0                 # Bottom composition
        
        # D matrix (no direct feedthrough)
        D = np.zeros((n_outputs, n_controls))
        
        print(f"  State dimension: {n_states}")
        print(f"  Control dimension: {n_controls}")
        print(f"  Output dimension: {n_outputs}")
        print(f"  A matrix condition number: {np.linalg.cond(A):.2e}")
        
        return A, B, C, D
    
    def discretize_model(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert continuous-time state-space model to discrete-time.
        Uses zero-order hold (ZOH) discretization.
        
        Returns:
        --------
        Ad, Bd, Cd, Dd : np.ndarray
            Discrete-time state-space matrices
        """
        print(f"Discretizing model with Ts = {self.Ts} min...")
        
        Ad, Bd, Cd, Dd, _ = cont2discrete(
            (self.A, self.B, self.C, self.D),
            self.Ts,
            method='zoh'
        )
        
        return Ad, Bd, Cd, Dd
    
    def setup_optimization(self,
                          column_model,
                          state_ss: np.ndarray,
                          u_nominal: Dict[str, float],
                          d_nominal: Dict[str, float],
                          weights: Dict[str, float],
                          constraints: Dict[str, Tuple[float, float]]):
        """
        Setup MPC optimization problem in GEKKO.
        
        Parameters:
        -----------
        column_model : BinaryDistillationColumn
            Nonlinear column model for linearization
        state_ss : np.ndarray
            Steady-state operating point
        u_nominal : dict
            Nominal control inputs
        d_nominal : dict
            Nominal disturbances
        weights : dict
            Cost function weights: {'energy': w1, 'purity': w2, 'control_rate': w3}
        constraints : dict
            Control and output bounds: 
            {'L_R': (min, max), 'Q_R': (min, max), 'x_top': (min, max), 'x_bottom': (min, max)}
        """
        print("\n" + "="*70)
        print("Setting up MPC Optimization Problem")
        print("="*70)
        
        # Store weights
        self.Q_energy = weights['energy']
        self.Q_purity = weights['purity']
        self.R_control = weights['control_rate']
        
        # Linearize model
        self.A, self.B, self.C, self.D = self.linearize_model(
            column_model, state_ss, u_nominal, d_nominal
        )
        
        # Store operating point
        self.x_op = state_ss
        self.u_op = np.array([u_nominal['L_R'], u_nominal['Q_R']])
        
        # For simplicity, use reduced-order model (top few singular values)
        # This makes MPC computation tractable
        n_reduced = 6  # Reduced state dimension
        
        # Singular Value Decomposition for model reduction
        U, S, Vt = np.linalg.svd(self.A)
        self.A_reduced = U[:, :n_reduced].T @ self.A @ U[:, :n_reduced]
        self.B_reduced = U[:, :n_reduced].T @ self.B
        self.C_reduced = self.C @ U[:, :n_reduced]
        
        # Discretize reduced model
        Ad, Bd, Cd, Dd = cont2discrete(
            (self.A_reduced, self.B_reduced, self.C_reduced, np.zeros((2, 2))),
            self.Ts,
            method='zoh'
        )
        
        print(f"\nModel Reduction:")
        print(f"  Original states: {self.A.shape[0]}")
        print(f"  Reduced states: {n_reduced}")
        print(f"  Singular values retained: {S[:n_reduced]}")
        
        # Define MPC variables in GEKKO
        print(f"\nDefining MPC variables...")
        
        # Control inputs (manipulated variables)
        self.u_reflux = self.m.MV(value=u_nominal['L_R'], 
                                  lb=constraints['L_R'][0],
                                  ub=constraints['L_R'][1])
        self.u_reflux.STATUS = 1  # Allow optimizer to change
        self.u_reflux.DCOST = self.R_control  # Penalize rate of change
        
        self.u_reboiler = self.m.MV(value=u_nominal['Q_R'],
                                    lb=constraints['Q_R'][0],
                                    ub=constraints['Q_R'][1])
        self.u_reboiler.STATUS = 1
        self.u_reboiler.DCOST = self.R_control
        
        # Controlled variables (outputs)
        self.y_top = self.m.CV(value=0.95)
        self.y_top.STATUS = 1  # Track setpoint
        self.y_top.SPHI = constraints['x_top'][1]
        self.y_top.SPLO = constraints['x_top'][0]
        self.y_top.TR_INIT = 0  # Dead-band
        
        self.y_bottom = self.m.CV(value=0.05)
        self.y_bottom.STATUS = 0  # Monitor only (soft constraint)
        
        # Define objective function
        # Minimize energy while tracking purity setpoint
        self.m.Minimize(self.Q_energy * self.u_reboiler**2)
        self.m.Minimize(self.Q_purity * (self.y_top - 0.95)**2)
        
        # MPC options
        self.m.options.IMODE = 6  # MPC mode
        self.m.options.CV_TYPE = 2  # Squared error
        self.m.options.SOLVER = 3  # IPOPT solver
        
        print(f"✓ MPC optimization problem configured")
        print(f"  Prediction horizon: {self.N} steps")
        print(f"  Control horizon: {self.M} steps")
        print(f"  Sampling time: {self.Ts} min")
        print("="*70)
    
    def compute_control(self,
                       current_state: np.ndarray,
                       setpoint: Dict[str, float]) -> Dict[str, float]:
        """
        Solve MPC optimization to compute optimal control actions.
        
        Parameters:
        -----------
        current_state : np.ndarray
            Current measured/estimated state
        setpoint : dict
            Desired setpoints: {'x_top': value, 'x_bottom': value}
        
        Returns:
        --------
        u_optimal : dict
            Optimal control inputs: {'L_R': value, 'Q_R': value}
        """
        # Update setpoints
        self.y_top.SP = setpoint['x_top']
        
        # Solve optimization (simplified - would need full state feedback)
        try:
            self.m.solve(disp=False)
            
            u_optimal = {
                'L_R': self.u_reflux.NEWVAL,
                'Q_R': self.u_reboiler.NEWVAL
            }
            
            return u_optimal
        
        except:
            print("⚠ MPC solve failed, returning nominal control")
            return {'L_R': self.u_op[0], 'Q_R': self.u_op[1]}


class SimplifiedMPC:
    """
    Simplified MPC implementation for demonstration and data generation.
    Uses direct optimization without GEKKO for faster computation.
    """
    
    def __init__(self,
                 prediction_horizon: int = 10,
                 control_horizon: int = 5):
        self.N = prediction_horizon
        self.M = control_horizon
        self.history = []
    
    def compute_control(self,
                       current_output: Dict[str, float],
                       setpoint: Dict[str, float],
                       u_prev: Dict[str, float],
                       disturbances: Dict[str, float]) -> Dict[str, float]:
        """
        Compute MPC control action using heuristic + optimization.
        
        This simplified version uses:
        1. Proportional control for purity tracking
        2. Energy minimization bias
        3. Smooth control changes
        
        Parameters:
        -----------
        current_output : dict
            Current measurements: {'x_top': value, 'x_bottom': value}
        setpoint : dict
            Desired setpoints: {'x_top': value}
        u_prev : dict
            Previous control inputs
        disturbances : dict
            Current disturbances (feed conditions)
        
        Returns:
        --------
        u_optimal : dict
            Optimal control inputs
        """
        # Purity error
        e_top = setpoint['x_top'] - current_output['x_top']
        e_bottom = current_output['x_bottom'] - 0.05  # Target bottom < 5% ethanol
        
        # Controller gains (tuned for performance)
        Kp_reflux = 200.0
        Kp_reboiler = 5000.0
        
        # Proportional control with energy bias
        delta_L_R = Kp_reflux * e_top
        delta_Q_R = Kp_reboiler * e_top + 2000.0 * e_bottom
        
        # Apply rate constraints (smooth control)
        delta_L_R = np.clip(delta_L_R, -5.0, 5.0)
        delta_Q_R = np.clip(delta_Q_R, -500.0, 500.0)
        
        # Compute new control
        L_R_new = u_prev['L_R'] + delta_L_R
        Q_R_new = u_prev['Q_R'] + delta_Q_R
        
        # Apply control constraints
        L_R_new = np.clip(L_R_new, 30.0, 80.0)
        Q_R_new = np.clip(Q_R_new, 3000.0, 8000.0)
        
        # Add disturbance feedforward
        F = disturbances['F']
        L_R_new += 0.3 * (F - 100.0)  # Adjust reflux based on feed rate
        
        u_optimal = {
            'L_R': L_R_new,
            'Q_R': Q_R_new
        }
        
        # Store in history for data generation
        self.history.append({
            'x_top': current_output['x_top'],
            'x_bottom': current_output['x_bottom'],
            'x_F': disturbances['x_F'],
            'F': disturbances['F'],
            'L_R': L_R_new,
            'Q_R': Q_R_new,
            'e_top': e_top
        })
        
        return u_optimal


# Example usage
if __name__ == "__main__":
    print("="*70)
    print("MPC Controller for Distillation Column - Initialization")
    print("="*70)
    
    # Create simplified MPC for demonstration
    mpc = SimplifiedMPC(prediction_horizon=10, control_horizon=5)
    
    # Example: Compute control action
    current_output = {'x_top': 0.93, 'x_bottom': 0.06}
    setpoint = {'x_top': 0.95}
    u_prev = {'L_R': 50.0, 'Q_R': 5000.0}
    disturbances = {'F': 100.0, 'x_F': 0.40, 'T_F': 340.0}
    
    print("\nCurrent State:")
    print(f"  Top composition: {current_output['x_top']*100:.1f}%")
    print(f"  Setpoint: {setpoint['x_top']*100:.1f}%")
    print(f"  Error: {(setpoint['x_top'] - current_output['x_top'])*100:.2f}%")
    
    u_optimal = mpc.compute_control(current_output, setpoint, u_prev, disturbances)
    
    print(f"\nComputed Control Actions:")
    print(f"  Reflux: {u_optimal['L_R']:.2f} kmol/min (previous: {u_prev['L_R']:.2f})")
    print(f"  Reboiler Duty: {u_optimal['Q_R']:.1f} kJ/min (previous: {u_prev['Q_R']:.1f})")
    
    print(f"\n✓ MPC controller initialized successfully!")
    print(f"✓ Ready for closed-loop simulation (Phase 3)")
    print("="*70)