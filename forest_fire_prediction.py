"""
Forest Fire Prediction - Implementation Plan
==========================================
Authors: Devansha Kaur and Tanisha Bohra python3 -c "import numpy; print('Numpy installed successfully')"a
CS 439 Project

This code provides a structured implementation for the forest fire prediction project,
following the proposal's outlined methodology.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           roc_auc_score, r2_score, mean_squared_error, mean_absolute_error)

# 1. Data Loading and Exploration
def load_and_explore_data():
    """Load the UCI Forest Fires dataset and perform initial exploration."""
    # Load data
    data = pd.read_csv('forestfires.csv')
    
    # Display basic information
    print("Dataset Shape:", data.shape)
    print("\nData Types:")
    print(data.dtypes)
    print("\nSummary Statistics:")
    print(data.describe())
    
    # Check for missing values
    print("\nMissing Values:")
    print(data.isnull().sum())
    
    return data

# 2. Data Visualization
def visualize_data(data):
    """Create exploratory visualizations of the dataset."""
    # Distribution of target variable
    plt.figure(figsize=(10, 6))
    plt.hist(data['area'], bins=50)
    plt.title('Distribution of Burned Area')
    plt.xlabel('Area (hectares)')
    plt.ylabel('Frequency')
    plt.savefig('area_distribution.png')
    
    # Log transformation of target (often needed for fire data)
    plt.figure(figsize=(10, 6))
    plt.hist(np.log1p(data['area']), bins=50)
    plt.title('Distribution of Log-Transformed Burned Area')
    plt.xlabel('Log(Area + 1)')
    plt.ylabel('Frequency')
    plt.savefig('log_area_distribution.png')
    
    # Correlation heatmap
    plt.figure(figsize=(12, 10))
    numerical_data = data.select_dtypes(include=[np.number])
    sns.heatmap(numerical_data.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.savefig('correlation_heatmap.png')
    
    # Monthly distribution of fires
    plt.figure(figsize=(10, 6))
    month_order = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 
                   'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    month_counts = data['month'].value_counts().reindex(month_order)
    sns.barplot(x=month_counts.index, y=month_counts.values)
    plt.title('Number of Fire Incidents by Month')
    plt.xlabel('Month')
    plt.ylabel('Count')
    plt.savefig('monthly_distribution.png')
    
    # Analyze fires by meteorological conditions
    plt.figure(figsize=(15, 10))
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    sns.scatterplot(data=data, x='temp', y='area', ax=axes[0, 0])
    axes[0, 0].set_title('Temperature vs Burned Area')
    
    sns.scatterplot(data=data, x='RH', y='area', ax=axes[0, 1])
    axes[0, 1].set_title('Relative Humidity vs Burned Area')
    
    sns.scatterplot(data=data, x='wind', y='area', ax=axes[1, 0])
    axes[1, 0].set_title('Wind Speed vs Burned Area')
    
    sns.scatterplot(data=data, x='rain', y='area', ax=axes[1, 1])
    axes[1, 1].set_title('Rainfall vs Burned Area')
    
    plt.tight_layout()
    plt.savefig('weather_vs_area.png')

# 3. Feature Engineering
def engineer_features(data):
    """Create new features that might improve model performance."""
    # Create a copy to avoid modifying original
    df = data.copy()
    
    # Convert month to numerical (seasonal) representation
    month_to_num = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    df['month_num'] = df['month'].map(month_to_num)
    
    # Create season feature
    df['season'] = pd.cut(
        df['month_num'], 
        bins=[0, 3, 6, 9, 12], 
        labels=['winter', 'spring', 'summer', 'fall'],
        include_lowest=True
    )
    
    # Create a binary target for classification
    # Fires larger than 1 hectare are considered significant
    df['large_fire'] = (df['area'] >= 1.0).astype(int)
    
    # Create log-transformed target for regression
    df['log_area'] = np.log1p(df['area'])
    
    # Create interaction features
    df['temp_wind'] = df['temp'] * df['wind']  # Hot and windy conditions
    df['RH_temp'] = df['RH'] * df['temp']      # Humidity-temperature interaction
    
    # Create drought index (simplified)
    # Lower values indicate higher drought risk
    df['drought_index'] = df['RH'] + df['rain'] * 10 - df['temp']
    
    # One-hot encode categorical variables
    df = pd.get_dummies(df, columns=['day', 'season'], drop_first=True)
    
    return df

# 4. Data Preprocessing
def preprocess_data(df, task='classification'):
    """Prepare data for model training."""
    # Define features and target
    if task == 'classification':
        y = df['large_fire']
    else:  # regression
        y = df['log_area']
    
    # Select numerical features
    numerical_features = ['temp', 'RH', 'wind', 'rain', 'FFMC', 'DMC', 'DC', 'ISI', 
                          'month_num', 'temp_wind', 'RH_temp', 'drought_index']
    
    # Find categorical features (one-hot encoded)
    categorical_features = [col for col in df.columns if col.startswith(('day_', 'season_'))]
    
    # Combine all features
    features = numerical_features + categorical_features
    X = df[features]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale numerical features
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[numerical_features] = scaler.fit_transform(X_train[numerical_features])
    X_test_scaled[numerical_features] = scaler.transform(X_test[numerical_features])
    
    return X_train_scaled, X_test_scaled, y_train, y_test

# 5. Model Training and Evaluation - Classification
def train_classification_models(X_train, X_test, y_train, y_test):
    """Train and evaluate classification models."""
    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42)
    }
    
    # Simple parameter grids for each model
    param_grids = {
        'Logistic Regression': {'C': [0.01, 0.1, 1, 10, 100]},
        'Random Forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }
    }
    
    results = {}
    feature_importance = {}
    
    for name, model in models.items():
        # Perform grid search with cross-validation
        grid_search = GridSearchCV(
            model, param_grids[name], cv=5, scoring='f1', n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        # Get best model
        best_model = grid_search.best_estimator_
        
        # Make predictions
        y_pred = best_model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
        }
        
        # Add ROC-AUC if the model can predict probabilities
        if hasattr(best_model, 'predict_proba'):
            y_prob = best_model.predict_proba(X_test)[:, 1]
            metrics['roc_auc'] = roc_auc_score(y_test, y_prob)
        
        results[name] = {
            'best_params': grid_search.best_params_,
            'metrics': metrics
        }
        
        # Get feature importances if available
        if hasattr(best_model, 'feature_importances_'):
            feature_importance[name] = {
                'features': X_train.columns,
                'importances': best_model.feature_importances_
            }
        elif hasattr(best_model, 'coef_'):
            feature_importance[name] = {
                'features': X_train.columns,
                'importances': best_model.coef_[0]
            }
    
    return results, feature_importance

# 6. Model Training and Evaluation - Regression
def train_regression_models(X_train, X_test, y_train, y_test):
    """Train and evaluate regression models."""
    # Initialize models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(random_state=42)
    }
    
    # Simple parameter grids for each model
    param_grids = {
        'Linear Regression': {},  # No hyperparameters to tune
        'Random Forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }
    }
    
    results = {}
    feature_importance = {}
    
    for name, model in models.items():
        # For Linear Regression, skip grid search
        if name == 'Linear Regression':
            best_model = model
            best_model.fit(X_train, y_train)
        else:
            # Perform grid search with cross-validation
            grid_search = GridSearchCV(
                model, param_grids[name], cv=5, scoring='neg_mean_squared_error', n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
        
        # Make predictions
        y_pred = best_model.predict(X_test)
        
        # Transform predictions back to original scale for interpretability
        y_test_original = np.expm1(y_test)
        y_pred_original = np.expm1(y_pred)
        
        # Calculate metrics
        metrics = {
            'r2': r2_score(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse_original': np.sqrt(mean_squared_error(y_test_original, y_pred_original)),
            'mae_original': mean_absolute_error(y_test_original, y_pred_original)
        }
        
        results[name] = {
            'best_params': grid_search.best_params_ if name != 'Linear Regression' else {},
            'metrics': metrics
        }
        
        # Get feature importances if available
        if hasattr(best_model, 'feature_importances_'):
            feature_importance[name] = {
                'features': X_train.columns,
                'importances': best_model.feature_importances_
            }
        elif hasattr(best_model, 'coef_'):
            feature_importance[name] = {
                'features': X_train.columns,
                'importances': best_model.coef_
            }
    
    return results, feature_importance

# 7. Visualize Results
def visualize_results(classification_results, regression_results, 
                     classification_importance, regression_importance):
    """Visualize model performance and feature importance."""
    # Classification metrics comparison
    plt.figure(figsize=(10, 6))
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    for name in classification_results.keys():
        values = [classification_results[name]['metrics'][metric] for metric in metrics]
        plt.bar(metrics, values, alpha=0.7, label=name)
    
    plt.title('Classification Model Performance')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig('classification_metrics.png')
    
    # Regression metrics comparison
    plt.figure(figsize=(10, 6))
    
    metrics = ['r2', 'rmse', 'mae']
    for name in regression_results.keys():
        values = [regression_results[name]['metrics'][metric] for metric in metrics]
        plt.bar([f"{metric}_{name}" for metric in metrics], values, alpha=0.7)
    
    plt.title('Regression Model Performance')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('regression_metrics.png')
    
    # Feature importance for best classification model
    best_clf_name = max(classification_results, 
                        key=lambda k: classification_results[k]['metrics']['f1'])
    
    if best_clf_name in classification_importance:
        plt.figure(figsize=(12, 8))
        features = classification_importance[best_clf_name]['features']
        importances = classification_importance[best_clf_name]['importances']
        
        # Sort by importance
        indices = np.argsort(importances)[::-1]
        
        plt.barh(range(len(indices)), importances[indices])
        plt.yticks(range(len(indices)), [features[i] for i in indices])
        plt.title(f'Feature Importance - {best_clf_name} (Classification)')
        plt.tight_layout()
        plt.savefig('classification_feature_importance.png')
    
    # Feature importance for best regression model
    best_reg_name = max(regression_results, 
                       key=lambda k: regression_results[k]['metrics']['r2'])
    
    if best_reg_name in regression_importance:
        plt.figure(figsize=(12, 8))
        features = regression_importance[best_reg_name]['features']
        importances = np.abs(regression_importance[best_reg_name]['importances'])
        
        # Sort by importance
        indices = np.argsort(importances)[::-1]
        
        plt.barh(range(len(indices)), importances[indices])
        plt.yticks(range(len(indices)), [features[i] for i in indices])
        plt.title(f'Feature Importance - {best_reg_name} (Regression)')
        plt.tight_layout()
        plt.savefig('regression_feature_importance.png')

# 8. Main function to run the entire analysis
def main():
    # Load and explore data
    data = load_and_explore_data()
    
    # Visualize raw data
    visualize_data(data)
    
    # Engineer features
    df_engineered = engineer_features(data)
    
    # Classification task
    print("\n--- Classification Task: Predicting Fire Occurrence ---")
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = preprocess_data(
        df_engineered, task='classification'
    )
    clf_results, clf_importance = train_classification_models(
        X_train_clf, X_test_clf, y_train_clf, y_test_clf
    )
    
    # Print classification results
    for name, result in clf_results.items():
        print(f"\n{name}:")
        print(f"Best Parameters: {result['best_params']}")
        print(f"Metrics: {result['metrics']}")
    
    # Regression task
    print("\n--- Regression Task: Predicting Fire Size ---")
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = preprocess_data(
        df_engineered, task='regression'
    )
    reg_results, reg_importance = train_regression_models(
        X_train_reg, X_test_reg, y_train_reg, y_test_reg
    )
    
    # Print regression results
    for name, result in reg_results.items():
        print(f"\n{name}:")
        print(f"Best Parameters: {result['best_params']}")
        print(f"Metrics: {result['metrics']}")
    
    # Visualize results
    visualize_results(clf_results, reg_results, clf_importance, reg_importance)
    
    # Save final results to CSV
    pd.DataFrame({
        'Model': list(clf_results.keys()) + list(reg_results.keys()),
        'Task': ['Classification'] * len(clf_results) + ['Regression'] * len(reg_results),
        'Best Parameters': [str(r['best_params']) for r in clf_results.values()] + 
                          [str(r['best_params']) for r in reg_results.values()],
        'Metrics': [str(r['metrics']) for r in clf_results.values()] + 
                  [str(r['metrics']) for r in reg_results.values()]
    }).to_csv('model_results.csv', index=False)
    
    print("\nAnalysis complete. Results saved to CSV and visualizations saved as PNG files.")

if __name__ == "__main__":
    main()