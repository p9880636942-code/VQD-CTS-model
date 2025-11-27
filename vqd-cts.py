
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class VQD_CTS_Model_Enhanced:
    """
    Enhanced VQD-CTS Prediction Model for real-world dataset
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.performance_metrics = {}
        
    def load_and_preprocess_data(self, file_path):
        """
        Load and preprocess the actual VQD-CTS dataset
        """
        # Load the dataset
        df = pd.read_csv(file_path)
        
        print(f"Dataset loaded: {len(df)} records with {len(df.columns)} features")
        print(f"Columns: {df.columns.tolist()}")
        
        # Check for missing values
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            print("Missing values found:")
            print(missing_values[missing_values > 0])
            # Fill missing values with median for numerical columns
            for column in df.select_dtypes(include=[np.number]).columns:
                df[column].fillna(df[column].median(), inplace=True)
        
        # Basic dataset statistics
        print(f"\nDataset Statistics:")
        print(f"Cost-to-Serve range: ${df['cost_to_serve'].min():.0f} - ${df['cost_to_serve'].max():.0f}")
        print(f"Project Context distribution:")
        print(df['project_context'].value_counts())
        
        return df
    
    def engineer_features(self, df):
        """
        Perform feature engineering on the real dataset
        """
        print("Performing feature engineering...")
        
        # Create interaction terms based on domain knowledge
        df['complexity_defect_interaction'] = df['feature_complexity'] * df['defect_density']
        df['velocity_quality_ratio'] = df['story_points_completed'] / (df['defect_density'] + 0.1)
        
        # Team productivity metrics
        if 'team_size' in df.columns and 'story_points_completed' in df.columns:
            df['productivity_per_dev'] = df['story_points_completed'] / df['team_size']
        
        # Cost efficiency metrics
        df['cost_per_story_point'] = df['cost_to_serve'] / df['story_points_completed']
        
        # Quality efficiency
        df['quality_efficiency'] = df['test_coverage'] / (df['defect_density'] + 0.1)
        
        # Apply non-linear transformations for key features
        df['defect_density_sq'] = df['defect_density'] ** 2
        if 'feature_complexity' in df.columns:
            df['complexity_log'] = np.log(df['feature_complexity'] + 1)
        
        # Cycle time efficiency
        if 'cycle_time' in df.columns and 'story_points_completed' in df.columns:
            df['velocity_efficiency'] = df['story_points_completed'] / df['cycle_time']
        
        print(f"Added {len([col for col in df.columns if col not in ['cost_to_serve', 'project_context']])} engineered features")
        
        return df
    
    def prepare_features(self, df):
        """
        Prepare features for model training from actual dataset
        """
        # Select feature columns (adjust based on your actual dataset)
        base_features = [
            'story_points_completed', 'commits_per_dev', 'cycle_time', 
            'deployment_frequency', 'lead_time', 'defect_density', 
            'escaped_defects', 'test_coverage', 'technical_debt_ratio', 
            'code_smell_density', 'onboarding_hours', 'developer_satisfaction',
            'tool_efficiency', 'context_switch_frequency', 'feedback_loop_time',
            'infrastructure_cost_share', 'average_hourly_rate', 'feature_complexity',
            'team_size', 'domain_complexity'
        ]
        
        # Only use features that exist in the dataset
        available_features = [f for f in base_features if f in df.columns]
        
        # Add engineered features
        engineered_features = [
            'complexity_defect_interaction', 'velocity_quality_ratio',
            'defect_density_sq', 'quality_efficiency'
        ]
        
        available_engineered = [f for f in engineered_features if f in df.columns]
        
        # Combine all available features
        feature_columns = available_features + available_engineered
        
        print(f"Using {len(feature_columns)} features for modeling")
        print(f"Features: {feature_columns}")
        
        # Convert categorical project context to dummy variables
        context_dummies = pd.get_dummies(df['project_context'], prefix='context')
        
        # Combine all features
        X = pd.concat([df[feature_columns], context_dummies], axis=1)
        y = df['cost_to_serve']
        
        self.feature_names = X.columns.tolist()
        
        return X, y
    
    def train_model(self, df):
        """
        Train the Random Forest model on actual data
        """
        # Prepare features
        X, y = self.prepare_features(df)
        
        # Split data (80% train, 15% validation, 5% test)
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.05, random_state=42, stratify=df['project_context']
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.15/0.95, random_state=42, 
            stratify=df.loc[X_temp.index]['project_context']
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Initialize and train Random Forest
        self.model = RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Store data splits for evaluation
        self.data_splits = {
            'X_train': X_train_scaled, 'y_train': y_train,
            'X_val': X_val_scaled, 'y_val': y_val,
            'X_test': X_test_scaled, 'y_test': y_test,
            'feature_names': self.feature_names,
            'train_indices': X_train.index,
            'val_indices': X_val.index,
            'test_indices': X_test.index
        }
        
        print(f"Model trained successfully:")
        print(f"  Training set: {len(y_train)} samples")
        print(f"  Validation set: {len(y_val)} samples")
        print(f"  Test set: {len(y_test)} samples")
        
        return self.data_splits
    
    def evaluate_model(self):
        """
        Comprehensive model evaluation
        """
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
        
        splits = self.data_splits
        results = {}
        
        for split_name in ['train', 'val', 'test']:
            if split_name == 'train':
                X, y = splits['X_train'], splits['y_train']
            elif split_name == 'val':
                X, y = splits['X_val'], splits['y_val']
            else:
                X, y = splits['X_test'], splits['y_test']
            
            y_pred = self.model.predict(X)
            
            # Predictive accuracy metrics
            mae = mean_absolute_error(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            r2 = r2_score(y, y_pred)
            mape = np.mean(np.abs((y - y_pred) / y)) * 100
            
            # Business impact metrics
            within_15_pct = np.mean(np.abs((y - y_pred) / y) <= 0.15) * 100
            
            # High-cost identification (top 20%)
            high_cost_threshold = np.percentile(y, 80)
            high_cost_mask = y >= high_cost_threshold
            high_cost_recall = np.sum((y_pred >= high_cost_threshold) & high_cost_mask) / np.sum(high_cost_mask)
            
            results[split_name] = {
                'R² Score': r2,
                'MAE': mae,
                'RMSE': rmse,
                'MAPE': mape,
                'Cost Prediction Accuracy (within 15%)': within_15_pct,
                'High-Cost Identification Recall': high_cost_recall,
                'predictions': y_pred,
                'actuals': y
            }
        
        self.performance_metrics = results
        return results
    
    def plot_feature_importance(self, top_n=15):
        """
        Generate feature importance plot
        """
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        importances = self.model.feature_importances_
        feature_names = self.data_splits['feature_names']
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=True).tail(top_n)
        
        # Plot
        plt.figure(figsize=(12, 8))
        colors = plt.cm.viridis(np.linspace(0, 1, len(importance_df)))
        bars = plt.barh(importance_df['feature'], importance_df['importance'], color=colors)
        
        # Add value labels on bars
        for bar in bars:
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height()/2, 
                    f'{width:.4f}', ha='left', va='center', fontweight='bold')
        
        plt.xlabel('Feature Importance Score', fontsize=12)
        plt.title('VQD-CTS Model: Feature Importance Analysis', fontsize=14, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        return importance_df
    
    def plot_predictions_vs_actuals(self):
        """
        Generate actual vs predicted scatter plot
        """
        if not self.performance_metrics:
            raise ValueError("Model must be evaluated first")
        
        test_results = self.performance_metrics['test']
        y_test = test_results['actuals']
        y_pred = test_results['predictions']
        
        plt.figure(figsize=(10, 8))
        
        # Create scatter plot with density coloring for large datasets
        if len(y_test) > 1000:
            plt.hexbin(y_test, y_pred, gridsize=50, cmap='Blues', alpha=0.7)
        else:
            plt.scatter(y_test, y_pred, alpha=0.6, s=50)
        
        # Perfect prediction line
        max_val = max(y_test.max(), y_pred.max())
        min_val = min(y_test.min(), y_pred.min())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
        
        plt.xlabel('Actual Cost-to-Serve ($)', fontsize=12)
        plt.ylabel('Predicted Cost-to-Serve ($)', fontsize=12)
        plt.title('VQD-CTS Model: Actual vs Predicted Values', fontsize=14, fontweight='bold')
        
        # Add performance metrics to plot
        r2 = test_results['R² Score']
        mape = test_results['MAPE']
        plt.text(0.05, 0.95, f'R² = {r2:.3f}\nMAPE = {mape:.1f}%', 
                transform=plt.gca().transAxes, fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        
        return np.corrcoef(y_test, y_pred)[0, 1]
    
    def plot_residuals(self):
        """
        Generate residual analysis plots
        """
        if not self.performance_metrics:
            raise ValueError("Model must be evaluated first")
        
        test_results = self.performance_metrics['test']
        y_test = test_results['actuals']
        y_pred = test_results['predictions']
        residuals = y_test - y_pred
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Residual scatter plot
        ax1.scatter(y_pred, residuals, alpha=0.6, s=50)
        ax1.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax1.set_xlabel('Predicted Cost-to-Serve ($)', fontsize=12)
        ax1.set_ylabel('Residuals ($)', fontsize=12)
        ax1.set_title('Residuals vs Predicted Values', fontsize=13, fontweight='bold')
        ax1.grid(alpha=0.3)
        
        # Residual distribution
        n, bins, patches = ax2.hist(residuals, bins=50, alpha=0.7, density=True, color='skyblue')
        ax2.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
        ax2.set_xlabel('Residuals ($)', fontsize=12)
        ax2.set_ylabel('Density', fontsize=12)
        ax2.set_title('Distribution of Prediction Residuals', fontsize=13, fontweight='bold')
        ax2.grid(alpha=0.3)
        
        # Add normal distribution overlay
        x = np.linspace(residuals.min(), residuals.max(), 100)
        pdf = stats.norm.pdf(x, residuals.mean(), residuals.std())
        ax2.plot(x, pdf, 'r-', linewidth=2, label='Normal Distribution')
        ax2.legend()
        
        # Add statistics
        ax2.text(0.05, 0.95, f'Mean: {residuals.mean():.2f}\nStd: {residuals.std():.2f}', 
                transform=ax2.transAxes, fontsize=11,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        
        return residuals
    
    def context_specific_performance(self, df):
        """
        Analyze performance across different project contexts
        """
        if not self.performance_metrics:
            raise ValueError("Model must be evaluated first")
        
        test_indices = self.data_splits['test_indices']
        test_df = df.loc[test_indices].copy()
        test_results = self.performance_metrics['test']
        y_pred = test_results['predictions']
        test_df['predicted_cts'] = y_pred
        
        context_performance = []
        
        for context in test_df['project_context'].unique():
            context_mask = test_df['project_context'] == context
            y_true_context = test_df.loc[context_mask, 'cost_to_serve']
            y_pred_context = test_df.loc[context_mask, 'predicted_cts']
            
            if len(y_true_context) > 1:  # Need at least 2 samples for R²
                r2 = r2_score(y_true_context, y_pred_context)
                mae = mean_absolute_error(y_true_context, y_pred_context)
                mape = np.mean(np.abs((y_true_context - y_pred_context) / y_true_context)) * 100
                sample_size = len(y_true_context)
                
                context_performance.append({
                    'Project Context': context,
                    'R²': r2,
                    'MAE': mae,
                    'MAPE': f'{mape:.1f}%',
                    'Sample Size': sample_size
                })
        
        return pd.DataFrame(context_performance)
    
    def generate_business_insights(self, df):
        """
        Generate actionable business insights from the model
        """
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        print("\n" + "="*60)
        print("BUSINESS INSIGHTS & RECOMMENDATIONS")
        print("="*60)
        
        # Feature importance insights
        importance_df = self.plot_feature_importance(top_n=10)
        top_features = importance_df.tail(5)['feature'].tolist()
        
        print(f"\nTop 5 Cost Drivers:")
        for i, feature in enumerate(reversed(top_features), 1):
            print(f"  {i}. {feature}")
        
        # Cost optimization opportunities
        print(f"\nCost Optimization Opportunities:")
        
        # Analyze defect density impact
        if 'defect_density' in df.columns:
            high_defect_threshold = df['defect_density'].quantile(0.75)
            high_defect_projects = df[df['defect_density'] > high_defect_threshold]
            if len(high_defect_projects) > 0:
                avg_cost_premium = (high_defect_projects['cost_to_serve'].mean() / 
                                  df['cost_to_serve'].mean() - 1) * 100
                print(f"  • High defect density (> {high_defect_threshold:.1f}) correlates with {avg_cost_premium:.1f}% higher costs")
        
        # Cycle time impact
        if 'cycle_time' in df.columns:
            slow_threshold = df['cycle_time'].quantile(0.75)
            slow_projects = df[df['cycle_time'] > slow_threshold]
            if len(slow_projects) > 0:
                cost_impact = (slow_projects['cost_to_serve'].mean() / 
                             df['cost_to_serve'].mean() - 1) * 100
                print(f"  • Long cycle times (> {slow_threshold:.1f} days) associated with {cost_impact:.1f}% cost increase")
        
        # Developer satisfaction impact
        if 'developer_satisfaction' in df.columns:
            low_satisfaction_threshold = df['developer_satisfaction'].quantile(0.25)
            low_sat_projects = df[df['developer_satisfaction'] < low_satisfaction_threshold]
            if len(low_sat_projects) > 0:
                satisfaction_impact = (low_sat_projects['cost_to_serve'].mean() / 
                                     df['cost_to_serve'].mean() - 1) * 100
                print(f"  • Low developer satisfaction (< {low_satisfaction_threshold:.1f}) links to {satisfaction_impact:.1f}% cost premium")
        
        return importance_df

# Main execution function for your dataset
def run_vqd_cts_analysis(csv_file_path):
    """
    Run complete VQD-CTS analysis on your dataset
    """
    print("=== VQD-CTS Prediction Model Analysis ===")
    print("Using actual project dataset\n")
    
    # Initialize enhanced model
    model = VQD_CTS_Model_Enhanced()
    
    # Load and preprocess data
    df = model.load_and_preprocess_data(csv_file_path)
    
    # Perform feature engineering
    df_engineered = model.engineer_features(df)
    
    # Train model
    print("\n" + "="*50)
    print("MODEL TRAINING")
    print("="*50)
    data_splits = model.train_model(df_engineered)
    
    # Evaluate model
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)
    results = model.evaluate_model()
    
    # Print performance results
    print("\nOverall Model Performance:")
    print("-" * 60)
    for split_name, metrics in results.items():
        print(f"{split_name.upper()} Set:")
        print(f"  R² Score: {metrics['R² Score']:.3f}")
        print(f"  MAE: ${metrics['MAE']:.2f}")
        print(f"  RMSE: ${metrics['RMSE']:.2f}")
        print(f"  MAPE: {metrics['MAPE']:.1f}%")
        print(f"  Cost Prediction Accuracy (within 15%): {metrics['Cost Prediction Accuracy (within 15%)']:.1f}%")
        print(f"  High-Cost Identification Recall: {metrics['High-Cost Identification Recall']:.3f}")
        print()
    
    # Generate visualizations
    print("\n" + "="*50)
    print("MODEL VISUALIZATIONS")
    print("="*50)
    
    # Feature importance
    importance_df = model.plot_feature_importance()
    
    # Predictions vs actuals
    correlation = model.plot_predictions_vs_actuals()
    print(f"Actual vs Predicted Correlation: {correlation:.3f}")
    
    # Residual analysis
    residuals = model.plot_residuals()
    print(f"Residual Analysis: Mean = ${residuals.mean():.2f}, Std = ${residuals.std():.2f}")
    
    # Context-specific performance
    print("\n" + "="*50)
    print("CONTEXT-SPECIFIC PERFORMANCE")
    print("="*50)
    context_perf = model.context_specific_performance(df_engineered)
    print(context_perf.to_string(index=False))
    
    # Business insights
    model.generate_business_insights(df_engineered)
    
    return model, df_engineered, results

# Run the analysis on your dataset
if __name__ == "__main__":
    # Replace with your actual CSV file path
    csv_file_path = "vqd_cts_dataset.csv"
    
    try:
        model, data, results = run_vqd_cts_analysis(csv_file_path)
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*60)
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        print("Please ensure the CSV file path is correct and the dataset format matches expectations.")