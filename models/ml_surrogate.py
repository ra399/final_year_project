"""
Machine Learning Surrogate Model with Explainable AI
====================================================
Author: BTP Student, Department of Chemical Engineering, IIT Patna
Project: XAI-MPC for Energy-Efficient Distillation Column Operation

Description:
-----------
This module implements:
1. Surrogate ML models (Random Forest, XGBoost) to approximate MPC control policy
2. SHAP (SHapley Additive exPlanations) for global and local interpretability
3. LIME (Local Interpretable Model-agnostic Explanations) for local explanations
4. Natural language explanation generation for operators

The surrogate model learns the mapping:
    State + Setpoint + Disturbances → Control Actions
    
This enables:
- Fast control computation (ms instead of seconds)
- Interpretable control decisions
- Operator trust and understanding

Installation:
------------
pip install scikit-learn xgboost shap lime pandas numpy matplotlib

References:
-----------
1. Lundberg & Lee (2017). "A Unified Approach to Interpreting Model Predictions"
2. Ribeiro et al. (2016). "Why Should I Trust You?"
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import shap
from lime import lime_tabular
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class SurrogateMLModel:
    """
    Machine Learning surrogate model for MPC control policy.
    
    The model learns to predict control actions (L_R, Q_R) from:
    - Current state (compositions, temperatures)
    - Setpoint (desired top composition)
    - Disturbances (feed conditions)
    
    Two models are trained:
    1. Random Forest: Ensemble of decision trees
    2. XGBoost: Gradient boosted trees
    """
    
    def __init__(self, model_type: str = 'xgboost'):
        """
        Initialize surrogate model.
        
        Parameters:
        -----------
        model_type : str
            'random_forest' or 'xgboost'
        """
        self.model_type = model_type
        self.model_L_R = None  # Model for reflux prediction
        self.model_Q_R = None  # Model for reboiler duty prediction
        self.scaler = StandardScaler()
        self.feature_names = None
        self.explainer_shap = None
        self.explainer_lime = None
        
    def prepare_features(self, data: pd.DataFrame) -> tuple:
        """
        Prepare feature matrix and target variables.
        
        Features:
        ---------
        - x_top: Current top composition
        - x_bottom: Current bottom composition
        - x_top_sp: Setpoint for top composition
        - purity_error: x_top_sp - x_top
        - F: Feed flowrate
        - x_F: Feed composition
        - T_F: Feed temperature
        
        Targets:
        --------
        - L_R: Reflux flowrate
        - Q_R: Reboiler duty
        
        Parameters:
        -----------
        data : pd.DataFrame
            Training data from MPC simulations
        
        Returns:
        --------
        X : np.ndarray
            Feature matrix
        y_L_R : np.ndarray
            Target for reflux
        y_Q_R : np.ndarray
            Target for reboiler duty
        """
        # Define features
        feature_cols = [
            'x_top', 'x_bottom', 'x_top_sp', 'F', 'x_F', 'T_F'
        ]
        
        # Compute derived features
        data['purity_error'] = data['x_top_sp'] - data['x_top']
        data['bottom_error'] = 0.05 - data['x_bottom']
        data['feed_disturbance'] = data['F'] - 100.0
        data['composition_disturbance'] = data['x_F'] - 0.40
        
        feature_cols.extend([
            'purity_error', 'bottom_error', 
            'feed_disturbance', 'composition_disturbance'
        ])
        
        self.feature_names = feature_cols
        
        X = data[feature_cols].values
        y_L_R = data['L_R'].values
        y_Q_R = data['Q_R'].values
        
        return X, y_L_R, y_Q_R
    
    def train(self, data: pd.DataFrame, test_size: float = 0.2):
        """
        Train surrogate models for both control outputs.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Training dataset from MPC simulations
        test_size : float
            Fraction of data for testing
        
        Returns:
        --------
        metrics : dict
            Performance metrics on test set
        """
        print("\n" + "="*70)
        print(f"Training {self.model_type.upper()} Surrogate Model")
        print("="*70)
        
        # Prepare data
        X, y_L_R, y_Q_R = self.prepare_features(data)
        
        print(f"\nDataset:")
        print(f"  Samples: {len(X)}")
        print(f"  Features: {X.shape[1]}")
        print(f"  Feature names: {self.feature_names}")
        
        # Split data
        X_train, X_test, y_L_R_train, y_L_R_test, y_Q_R_train, y_Q_R_test = \
            train_test_split(X, y_L_R, y_Q_R, test_size=test_size, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"\nTrain/Test Split:")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Testing samples: {len(X_test)}")
        
        # Train models
        print(f"\nTraining model for Reflux (L_R)...")
        self.model_L_R = self._create_model()
        self.model_L_R.fit(X_train_scaled, y_L_R_train)
        
        print(f"Training model for Reboiler Duty (Q_R)...")
        self.model_Q_R = self._create_model()
        self.model_Q_R.fit(X_train_scaled, y_Q_R_train)
        
        # Evaluate
        print(f"\nEvaluating model performance...")
        metrics = self._evaluate(X_test_scaled, y_L_R_test, y_Q_R_test)
        
        # Initialize explainers
        print(f"\nInitializing SHAP explainer...")
        self._init_shap_explainer(X_train_scaled)
        
        print(f"Initializing LIME explainer...")
        self._init_lime_explainer(X_train_scaled, self.feature_names)
        
        print(f"\n✓ Training complete!")
        print("="*70)
        
        return metrics
    
    def _create_model(self):
        """Create model instance based on model_type."""
        if self.model_type == 'random_forest':
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'xgboost':
            return xgb.XGBRegressor(
                n_estimators=100,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _evaluate(self, X_test, y_L_R_test, y_Q_R_test):
        """Evaluate model performance."""
        # Predictions
        y_L_R_pred = self.model_L_R.predict(X_test)
        y_Q_R_pred = self.model_Q_R.predict(X_test)
        
        # Metrics for L_R
        r2_L_R = r2_score(y_L_R_test, y_L_R_pred)
        mae_L_R = mean_absolute_error(y_L_R_test, y_L_R_pred)
        rmse_L_R = np.sqrt(mean_squared_error(y_L_R_test, y_L_R_pred))
        
        # Metrics for Q_R
        r2_Q_R = r2_score(y_Q_R_test, y_Q_R_pred)
        mae_Q_R = mean_absolute_error(y_Q_R_test, y_Q_R_pred)
        rmse_Q_R = np.sqrt(mean_squared_error(y_Q_R_test, y_Q_R_pred))
        
        metrics = {
            'L_R': {
                'R²': r2_L_R,
                'MAE': mae_L_R,
                'RMSE': rmse_L_R
            },
            'Q_R': {
                'R²': r2_Q_R,
                'MAE': mae_Q_R,
                'RMSE': rmse_Q_R
            }
        }
        
        print(f"\n{'='*70}")
        print("Model Performance Metrics")
        print(f"{'='*70}")
        print(f"\nReflux (L_R) Prediction:")
        print(f"  R² Score:  {r2_L_R:.4f}")
        print(f"  MAE:       {mae_L_R:.3f} kmol/min")
        print(f"  RMSE:      {rmse_L_R:.3f} kmol/min")
        
        print(f"\nReboiler Duty (Q_R) Prediction:")
        print(f"  R² Score:  {r2_Q_R:.4f}")
        print(f"  MAE:       {mae_Q_R:.1f} kJ/min")
        print(f"  RMSE:      {rmse_Q_R:.1f} kJ/min")
        
        return metrics
    
    def _init_shap_explainer(self, X_train):
        """Initialize SHAP explainer."""
        # Use TreeExplainer for tree-based models
        self.explainer_shap_L_R = shap.TreeExplainer(self.model_L_R)
        self.explainer_shap_Q_R = shap.TreeExplainer(self.model_Q_R)
    
    def _init_lime_explainer(self, X_train, feature_names):
        """Initialize LIME explainer."""
        self.explainer_lime_L_R = lime_tabular.LimeTabularExplainer(
            X_train,
            feature_names=feature_names,
            mode='regression',
            random_state=42
        )
        
        self.explainer_lime_Q_R = lime_tabular.LimeTabularExplainer(
            X_train,
            feature_names=feature_names,
            mode='regression',
            random_state=42
        )
    
    def predict(self, state: dict) -> dict:
        """
        Predict control actions for given state.
        
        Parameters:
        -----------
        state : dict
            Current state with keys matching feature names
        
        Returns:
        --------
        control : dict
            Predicted control actions: {'L_R': value, 'Q_R': value}
        """
        # Prepare input
        x = self._state_to_features(state)
        x_scaled = self.scaler.transform([x])
        
        # Predict
        L_R = self.model_L_R.predict(x_scaled)[0]
        Q_R = self.model_Q_R.predict(x_scaled)[0]
        
        return {'L_R': L_R, 'Q_R': Q_R}
    
    def _state_to_features(self, state: dict) -> np.ndarray:
        """Convert state dictionary to feature array."""
        features = []
        for name in self.feature_names:
            if name in state:
                features.append(state[name])
            elif name == 'purity_error':
                features.append(state.get('x_top_sp', 0.95) - state.get('x_top', 0.90))
            elif name == 'bottom_error':
                features.append(0.05 - state.get('x_bottom', 0.05))
            elif name == 'feed_disturbance':
                features.append(state.get('F', 100.0) - 100.0)
            elif name == 'composition_disturbance':
                features.append(state.get('x_F', 0.40) - 0.40)
            else:
                features.append(0.0)
        
        return np.array(features)
    
    def explain_with_shap(self, state: dict, save_plot: bool = True):
        """
        Generate SHAP explanations for a specific control decision.
        
        Parameters:
        -----------
        state : dict
            State to explain
        save_plot : bool
            Whether to save visualization
        
        Returns:
        --------
        explanation : dict
            SHAP values and feature contributions
        """
        print("\n" + "="*70)
        print("SHAP Explanation for Control Decision")
        print("="*70)
        
        # Prepare input
        x = self._state_to_features(state)
        x_scaled = self.scaler.transform([x])
        
        # Compute SHAP values
        shap_values_L_R = self.explainer_shap_L_R.shap_values(x_scaled)
        shap_values_Q_R = self.explainer_shap_Q_R.shap_values(x_scaled)
        
        # Get predictions
        pred_L_R = self.model_L_R.predict(x_scaled)[0]
        pred_Q_R = self.model_Q_R.predict(x_scaled)[0]
        
        # Expected values (baseline)
        base_L_R = self.explainer_shap_L_R.expected_value
        base_Q_R = self.explainer_shap_Q_R.expected_value
        
        print(f"\nPredicted Control Actions:")
        print(f"  Reflux (L_R): {pred_L_R:.2f} kmol/min")
        print(f"  Reboiler Duty (Q_R): {pred_Q_R:.1f} kJ/min")
        
        print(f"\nTop 5 Feature Contributions to Reflux:")
        indices_L_R = np.argsort(np.abs(shap_values_L_R[0]))[::-1][:5]
        for idx in indices_L_R:
            print(f"  {self.feature_names[idx]:25s}: {shap_values_L_R[0][idx]:+.3f}")
        
        print(f"\nTop 5 Feature Contributions to Reboiler Duty:")
        indices_Q_R = np.argsort(np.abs(shap_values_Q_R[0]))[::-1][:5]
        for idx in indices_Q_R:
            print(f"  {self.feature_names[idx]:25s}: {shap_values_Q_R[0][idx]:+.1f}")
        
        explanation = {
            'L_R': {
                'prediction': pred_L_R,
                'base_value': base_L_R,
                'shap_values': shap_values_L_R[0],
                'features': x,
                'feature_names': self.feature_names
            },
            'Q_R': {
                'prediction': pred_Q_R,
                'base_value': base_Q_R,
                'shap_values': shap_values_Q_R[0],
                'features': x,
                'feature_names': self.feature_names
            }
        }
        
        return explanation
    
    def explain_with_lime(self, state: dict, num_features: int = 5):
        """
        Generate LIME explanations for a specific control decision.
        
        Parameters:
        -----------
        state : dict
            State to explain
        num_features : int
            Number of top features to show
        
        Returns:
        --------
        explanation : dict
            LIME explanations
        """
        print("\n" + "="*70)
        print("LIME Explanation for Control Decision")
        print("="*70)
        
        # Prepare input
        x = self._state_to_features(state)
        x_scaled = self.scaler.transform([x])
        
        # Generate LIME explanations
        exp_L_R = self.explainer_lime_L_R.explain_instance(
            x_scaled[0],
            self.model_L_R.predict,
            num_features=num_features
        )
        
        exp_Q_R = self.explainer_lime_Q_R.explain_instance(
            x_scaled[0],
            self.model_Q_R.predict,
            num_features=num_features
        )
        
        print(f"\nLIME Explanation for Reflux (L_R):")
        for feature, weight in exp_L_R.as_list():
            print(f"  {feature:40s}: {weight:+.3f}")
        
        print(f"\nLIME Explanation for Reboiler Duty (Q_R):")
        for feature, weight in exp_Q_R.as_list():
            print(f"  {feature:40s}: {weight:+.1f}")
        
        return {'L_R': exp_L_R, 'Q_R': exp_Q_R}
    
    def generate_operator_explanation(self, state: dict) -> str:
        """
        Generate natural language explanation for operators.
        
        Parameters:
        -----------
        state : dict
            Current state
        
        Returns:
        --------
        explanation : str
            Human-readable explanation
        """
        # Get prediction
        control = self.predict(state)
        
        # Get SHAP explanation
        shap_exp = self.explain_with_shap(state, save_plot=False)
        
        # Build explanation
        explanation = f"""
╔══════════════════════════════════════════════════════════════════╗
║  CONTROL DECISION EXPLANATION FOR OPERATORS                      ║
╚══════════════════════════════════════════════════════════════════╝

Current Situation:
------------------
• Top composition: {state['x_top']*100:.1f}% ethanol
• Setpoint: {state.get('x_top_sp', 0.95)*100:.1f}% ethanol
• Error: {(state.get('x_top_sp', 0.95) - state['x_top'])*100:.2f}%
• Feed rate: {state.get('F', 100):.1f} kmol/min
• Feed composition: {state.get('x_F', 0.40)*100:.1f}% ethanol

Recommended Control Actions:
-----------------------------
• Reflux Rate: {control['L_R']:.1f} kmol/min
• Reboiler Duty: {control['Q_R']:.0f} kJ/min

Why These Actions?
------------------
"""
        
        # Analyze top contributing factors
        shap_L_R = shap_exp['L_R']['shap_values']
        top_idx_L_R = np.argsort(np.abs(shap_L_R))[::-1][0]
        top_feature_L_R = self.feature_names[top_idx_L_R]
        top_value_L_R = shap_L_R[top_idx_L_R]
        
        if 'error' in top_feature_L_R and top_value_L_R > 0:
            explanation += "• Product purity is below setpoint, so reflux is increased\n"
        elif 'error' in top_feature_L_R and top_value_L_R < 0:
            explanation += "• Product purity is above setpoint, so reflux is decreased\n"
        
        if 'F' in top_feature_L_R or 'feed' in top_feature_L_R:
            if state.get('F', 100) > 105:
                explanation += "• Feed rate is higher than normal, requiring more separation duty\n"
            elif state.get('F', 100) < 95:
                explanation += "• Feed rate is lower than normal, allowing reduced energy input\n"
        
        explanation += f"""
Key Factors Influencing Decision:
---------------------------------
"""
        
        # Top 3 factors
        for i in range(min(3, len(shap_L_R))):
            idx = np.argsort(np.abs(shap_L_R))[::-1][i]
            feature = self.feature_names[idx]
            impact = "increases" if shap_L_R[idx] > 0 else "decreases"
            explanation += f"• {feature}: {impact} control action\n"
        
        return explanation


# Example usage
if __name__ == "__main__":
    print("="*70)
    print("Surrogate ML Model with XAI - Test")
    print("="*70)
    
    # Load data (assuming it exists)
    try:
        data = pd.read_csv('mpc_training_data.csv')
    except:
        print("⚠ Warning: Training data not found. Generating synthetic data...")
        # Generate synthetic data for testing
        n = 1000
        data = pd.DataFrame({
            'x_top': np.random.uniform(0.90, 0.97, n),
            'x_bottom': np.random.uniform(0.03, 0.07, n),
            'x_top_sp': np.random.choice([0.93, 0.95], n),
            'F': np.random.uniform(90, 110, n),
            'x_F': np.random.uniform(0.35, 0.45, n),
            'T_F': np.random.uniform(335, 345, n),
            'L_R': np.random.uniform(45, 65, n),
            'Q_R': np.random.uniform(4500, 6500, n)
        })
    
    # Train model
    model = SurrogateMLModel(model_type='xgboost')
    metrics = model.train(data)
    
    # Test prediction and explanation
    test_state = {
        'x_top': 0.92,
        'x_bottom': 0.05,
        'x_top_sp': 0.95,
        'F': 105.0,
        'x_F': 0.42,
        'T_F': 340.0
    }
    
    print(f"\n{'='*70}")
    print("Testing Prediction and Explanation")
    print(f"{'='*70}")
    
    # Prediction
    control = model.predict(test_state)
    print(f"\nPredicted Control:")
    print(f"  L_R = {control['L_R']:.2f} kmol/min")
    print(f"  Q_R = {control['Q_R']:.1f} kJ/min")
    
    # Operator explanation
    explanation = model.generate_operator_explanation(test_state)
    print(explanation)
    
    print(f"\n✓ Surrogate model with XAI ready!")
    print(f"✓ Ready for dashboard integration (Phase 5)")
    print("="*70)