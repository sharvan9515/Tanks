# -*- coding: utf-8 -*-
"""
CLEAN 4-STAGE LIST PRICE MODEL WITH SEASONALITY
===============================================

Enhanced version with seasonality features from Created Date and Actual Ship Date.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load and split data chronologically"""
    print("Loading data with seasonality features...")
    #df = pd.read_excel('../data/Final_training_data2_with_listprice.xlsx')
    df = pd.read_excel('../new_training/Final_training_data2_with_ExtraCost_updated.xlsx')


    # Convert date columns
    df['Created Date'] = pd.to_datetime(df['Created Date'], errors='coerce')
    df['Actual Ship Date'] = pd.to_datetime(df['Actual Ship Date'], errors='coerce')

    valid_dates = df['Created Date'].notna()
    df_sorted = df[valid_dates].sort_values('Created Date').reset_index(drop=True)

    split_idx = 649
    train_data = df_sorted.iloc[:split_idx].copy()
    test_data = df_sorted.iloc[split_idx:].copy()

    print(f"Training: {len(train_data)}, Test: {len(test_data)}")
    return train_data, test_data

def add_seasonality_features(df):
    """Add comprehensive seasonality features from both dates"""
    df_processed = df.copy()

    # Created Date seasonality (order timing)
    df_processed['create_month'] = df_processed['Created Date'].dt.month
    df_processed['create_quarter'] = df_processed['Created Date'].dt.quarter
    df_processed['create_season'] = df_processed['create_month'].map({
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Fall', 10: 'Fall', 11: 'Fall'
    })


    # Actual Ship Date seasonality (completion timing)
    if df_processed['Actual Ship Date'].notna().any():
        df_processed['ship_month'] = df_processed['Actual Ship Date'].dt.month
        df_processed['ship_quarter'] = df_processed['Actual Ship Date'].dt.quarter
        df_processed['ship_season'] = df_processed['ship_month'].map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        })

        # Manufacturing/shipping capacity factors
        df_processed['ship_capacity_factor'] = df_processed['ship_month'].map({
            1: 1.0,  # January - Normal capacity
            2: 1.1,  # February - Higher capacity (pre-season rush)
            3: 1.2,  # March - Preparing for construction season
            4: 1.4,  # April - High demand/lower capacity





            5: 1.5,  # May - Peak demand
            6: 1.6,  # June - Highest demand
            7: 1.5,  # July - Still high
            8: 1.3,  # August - Moderating
            9: 1.2,  # September - Still elevated
            10: 1.1, # October - Normalizing
            11: 0.9, # November - Lower demand
            12: 0.8  # December - Lowest demand
        })

        # Cyclical encoding for months (captures cyclical nature)
        df_processed['ship_month_sin'] = np.sin(2 * np.pi * df_processed['ship_month'] / 12)
        df_processed['ship_month_cos'] = np.cos(2 * np.pi * df_processed['ship_month'] / 12)
    else:
        # Fill with default values if ship date not available
        df_processed['ship_month'] = df_processed['create_month']
        df_processed['ship_quarter'] = df_processed['create_quarter']
        df_processed['ship_season'] = df_processed['create_season']
        df_processed['ship_capacity_factor'] = 1.0
        df_processed['ship_month_sin'] = 0.0
        df_processed['ship_month_cos'] = 0.0

    # Cyclical encoding for created months
    df_processed['create_month_sin'] = np.sin(2 * np.pi * df_processed['create_month'] / 12)
    df_processed['create_month_cos'] = np.cos(2 * np.pi * df_processed['create_month'] / 12)

    # Lead time feature
    if 'Diff Create Date vs Actual Ship Date' in df_processed.columns:
        df_processed['lead_time'] = df_processed['Diff Create Date vs Actual Ship Date']

    return df_processed

def prepare_base_features(df):
    """Prepare comprehensive features for stages 1-3 including seasonality"""
    df_processed = df.copy()

    # Add seasonality features first
    df_processed = add_seasonality_features(df_processed)

    # Calculate engineered features
    df_processed['tank_volume'] = np.pi * (df_processed['DIAMETER']/2)**2 * df_processed['HEIGHT']
    df_processed['surface_area'] = 2 * np.pi * (df_processed['DIAMETER']/2) * df_processed['HEIGHT'] + 2 * np.pi * (df_processed['DIAMETER']/2)**2
    df_processed['volume_to_surface_ratio'] = df_processed['tank_volume'] / df_processed['surface_area']
    df_processed['aspect_ratio'] = df_processed['HEIGHT'] / df_processed['DIAMETER']

    # Material complexity
    material_complexity_map = {
        'A36': 1.0, 'A516 Gr 70': 1.2, 'A572 Gr 50': 1.3, 'Stainless Steel': 2.5
    }
    df_processed['material_complexity'] = df_processed['MATERIAL'].map(material_complexity_map).fillna(1.1)

    # Pressure complexity
    df_processed['pressure_complexity'] = np.where(
        df_processed['PRESSURE'] > 50, 2.0,
        np.where(df_processed['PRESSURE'] > 15, 1.5, 1.0)
    )


    return df_processed

def prepare_short_tank_features(df):
    """Prepare specialized features for short tanks WITHOUT seasonality"""
    df_processed = df.copy()

    # DO NOT add seasonality features - they cause overfitting to June outliers

    df_processed['tank_volume'] = np.pi * (df_processed['DIAMETER']/2)**2 * df_processed['HEIGHT']
    df_processed['aspect_ratio'] = df_processed['HEIGHT'] / np.maximum(df_processed['DIAMETER'], 1)

    df_processed['height_efficiency_penalty'] = np.where(
        df_processed['HEIGHT'] < 30, 1.4,
        np.where(df_processed['HEIGHT'] < 40, 1.2,
                np.where(df_processed['HEIGHT'] < 50, 1.1, 1.0))
    )

    df_processed['manufacturing_complexity'] = np.where(
        df_processed['HEIGHT'] < 35,
        1.3 + (35 - df_processed['HEIGHT']) * 0.05,
        1.1
    )

    standard_volume = 8000
    df_processed['volume_inefficiency'] = np.maximum(
        1.0, standard_volume / np.maximum(df_processed['tank_volume'], 1000)
    )

    return df_processed

def encode_features(X_train, X_test=None):
    """Encode categorical features"""
    encoders = {}
    X_train_encoded = X_train.copy().fillna(0)

    categorical_features = X_train_encoded.select_dtypes(include=['object']).columns

    for feature in categorical_features:
        encoders[feature] = LabelEncoder()
        X_train_encoded[feature] = encoders[feature].fit_transform(X_train_encoded[feature].astype(str))

    if X_test is not None:
        X_test_encoded = X_test.copy().fillna(0)
        for feature in categorical_features:
            if feature in X_test_encoded.columns:
                values = X_test_encoded[feature].astype(str)
                unseen = set(values) - set(encoders[feature].classes_)
                if unseen:
                    for val in unseen:
                        encoders[feature].classes_ = np.append(encoders[feature].classes_, val)
                X_test_encoded[feature] = encoders[feature].transform(values)
        return X_train_encoded, X_test_encoded, encoders

    return X_train_encoded, encoders

def train_4stage_model_with_seasonality(train_data):
    """Train all 4 stages with seasonality features"""
    print("Training 4-stage model with seasonality...")

    # Prepare base features
    train_processed = prepare_base_features(train_data)

    base_feature_columns = [
        # Original features
        'HEIGHT', 'DIAMETER', 'WEIGHT', 'PRESSURE', 'VACUUM', 'BULK',
        'MATERIAL', 'BOTTOM', 'PLANT', 'ORDERCLASS',
        'tank_volume', 'surface_area', 'volume_to_surface_ratio', 'aspect_ratio',
        'material_complexity', 'pressure_complexity',

        # Seasonality features
        'create_month', 'create_quarter', 'create_season',
        'ship_month', 'ship_quarter', 'ship_season',
        'ship_capacity_factor', 'create_month_sin', 'create_month_cos',
        'ship_month_sin', 'ship_month_cos'
    ]

    if 'list_price' in train_processed.columns:
        base_feature_columns.append('list_price')

    if 'lead_time' in train_processed.columns:
        base_feature_columns.append('lead_time')

    optional_features = ['WIND MPH', 'SEISMIC ZONE', 'PAINTCODE']
    for feature in optional_features:
        if feature in train_processed.columns:
            base_feature_columns.append(feature)

    available_base_features = [f for f in base_feature_columns if f in train_processed.columns]

    print(f"Base features with seasonality: {len(available_base_features)}")

    X_base = train_processed[available_base_features]
    y = train_processed['TotalCost']

    X_base_encoded, base_encoders = encode_features(X_base)

    models = {}

    # Stage 1: Conservative
    print("Training Stage 1 (Conservative) with seasonality...")
    models['stage1'] = XGBRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.1, random_state=42
    )
    models['stage1'].fit(X_base_encoded, y)

    # Stage 2: Adaptive
    print("Training Stage 2 (Adaptive) with seasonality...")
    models['stage2'] = XGBRegressor(
        n_estimators=300, max_depth=8, learning_rate=0.08,
        subsample=0.85, colsample_bytree=0.85,
        reg_alpha=0.05, reg_lambda=0.05, random_state=42
    )
    models['stage2'].fit(X_base_encoded, y)

    # Stage 3: Fine-tuning
    print("Training Stage 3 (Fine-tuning) with seasonality...")
    models['stage3'] = XGBRegressor(
        n_estimators=400, max_depth=10, learning_rate=0.1,
        subsample=0.9, colsample_bytree=0.9,
        reg_alpha=0.01, reg_lambda=0.01, random_state=42
    )
    models['stage3'].fit(X_base_encoded, y)

    # Stage 4: Short Tank Specialist (WITHOUT seasonal features to avoid overfitting)
    print("Training Stage 4 (Short Tank Specialist) WITHOUT seasonality...")
    short_tank_data = train_data[train_data['HEIGHT'] < 50.0].copy()

    if len(short_tank_data) > 0:
        short_processed = prepare_short_tank_features(short_tank_data)

        short_feature_columns = [
            # Core short tank features (NO seasonal features to avoid overfitting)
            'HEIGHT', 'DIAMETER', 'WEIGHT', 'PRESSURE', 'VACUUM',
            'MATERIAL', 'BOTTOM', 'PLANT', 'ORDERCLASS',
            'tank_volume', 'aspect_ratio', 'height_efficiency_penalty',
            'manufacturing_complexity', 'volume_inefficiency'
        ]

        if 'list_price' in short_processed.columns:
            short_feature_columns.append('list_price')

        available_short_features = [f for f in short_feature_columns if f in short_processed.columns]

        print(f"Short tank features (NO seasonality): {len(available_short_features)}")

        X_short = short_processed[available_short_features]
        y_short = short_processed['TotalCost']

        X_short_encoded, short_encoders = encode_features(X_short)

        models['stage4'] = XGBRegressor(
            n_estimators=500, max_depth=8, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.9,
            reg_alpha=0.1, reg_lambda=0.1, random_state=42
        )
        models['stage4'].fit(X_short_encoded, y_short)

        print(f"Short tank specialist trained on {len(short_tank_data)} samples")
    else:
        models['stage4'] = None
        short_encoders = {}
        available_short_features = []

    return models, base_encoders, short_encoders, available_base_features, available_short_features

def predict_4stage_with_seasonality(models, test_data, base_encoders, short_encoders, base_features, short_features):
    """Make predictions using 4-stage routing with seasonality"""
    print("Making 4-stage predictions with seasonality...")

    # Split by tank type
    short_mask = test_data['HEIGHT'] < 50.0
    normal_mask = test_data['HEIGHT'] >= 50.0

    short_count = short_mask.sum()
    normal_count = normal_mask.sum()

    print(f"Short tanks (4-stage): {short_count}")
    print(f"Normal tanks (3-stage): {normal_count}")

    predictions = np.zeros(len(test_data))

    # Predict normal tanks (stages 1-3)
    if normal_count > 0:
        normal_data = test_data[normal_mask]
        normal_processed = prepare_base_features(normal_data)

        X_normal = normal_processed[base_features].fillna(0)

        # Encode using base encoders
        for feature in X_normal.select_dtypes(include=['object']).columns:
            if feature in base_encoders:
                values = X_normal[feature].astype(str)
                unseen = set(values) - set(base_encoders[feature].classes_)
                if unseen:
                    for val in unseen:
                        base_encoders[feature].classes_ = np.append(base_encoders[feature].classes_, val)
                X_normal[feature] = base_encoders[feature].transform(values)

        # Ensemble prediction for normal tanks
        pred1 = models['stage1'].predict(X_normal)
        pred2 = models['stage2'].predict(X_normal)
        pred3 = models['stage3'].predict(X_normal)

        normal_predictions = 0.2 * pred1 + 0.3 * pred2 + 0.5 * pred3
        predictions[normal_mask] = normal_predictions

    # Predict short tanks (all 4 stages - with specialist)
    if short_count > 0 and models['stage4'] is not None:
        short_data = test_data[short_mask]

        # Base predictions (stages 1-3)
        short_base_processed = prepare_base_features(short_data)
        X_short_base = short_base_processed[base_features].fillna(0)

        # Encode for base features
        for feature in X_short_base.select_dtypes(include=['object']).columns:
            if feature in base_encoders:
                values = X_short_base[feature].astype(str)
                unseen = set(values) - set(base_encoders[feature].classes_)
                if unseen:
                    for val in unseen:
                        base_encoders[feature].classes_ = np.append(base_encoders[feature].classes_, val)
                X_short_base[feature] = base_encoders[feature].transform(values)

        pred1 = models['stage1'].predict(X_short_base)
        pred2 = models['stage2'].predict(X_short_base)
        pred3 = models['stage3'].predict(X_short_base)

        # Specialist prediction (stage 4 - without seasonal features)
        short_specialist_processed = prepare_short_tank_features(short_data)
        X_short_specialist = short_specialist_processed[short_features].fillna(0)

        # Encode for specialist features
        for feature in X_short_specialist.select_dtypes(include=['object']).columns:
            if feature in short_encoders:
                values = X_short_specialist[feature].astype(str)
                unseen = set(values) - set(short_encoders[feature].classes_)
                if unseen:
                    for val in unseen:
                        short_encoders[feature].classes_ = np.append(short_encoders[feature].classes_, val)
                X_short_specialist[feature] = short_encoders[feature].transform(values)

        pred4 = models['stage4'].predict(X_short_specialist)

        # Enhanced weighting for short tanks with specialist
        short_predictions = 0.1 * pred1 + 0.15 * pred2 + 0.25 * pred3 + 0.5 * pred4
        predictions[short_mask] = short_predictions

    return predictions

def get_feature_importance(models, base_features, short_features):
    """Extract and analyze feature importance"""
    print("\nFEATURE IMPORTANCE ANALYSIS:")
    print("=" * 30)

    # Get feature importance from each stage
    importance_data = []

    for stage_name, model in models.items():
        if model is not None:
            if stage_name == 'stage4':
                feature_names = short_features
            else:
                feature_names = base_features

            importances = model.feature_importances_

            for feature, importance in zip(feature_names, importances):
                importance_data.append({
                    'stage': stage_name,
                    'feature': feature,
                    'importance': importance
                })

    importance_df = pd.DataFrame(importance_data)

    # Average importance across stages for base features
    base_avg_importance = importance_df[importance_df['stage'].isin(['stage1', 'stage2', 'stage3'])].groupby('feature')['importance'].mean().sort_values(ascending=False)

    print("Top 10 Base Features (Stages 1-3):")
    for i, (feature, importance) in enumerate(base_avg_importance.head(10).items()):
        is_seasonal = any(season_word in feature.lower() for season_word in ['month', 'season', 'quarter', 'construction', 'capacity', 'sin', 'cos'])
        marker = "[SEASONAL]" if is_seasonal else ""
        print(f"{i+1:2d}. {feature:<35} {importance:.4f} {marker}")

    if models['stage4'] is not None:
        stage4_importance = importance_df[importance_df['stage'] == 'stage4'].set_index('feature')['importance'].sort_values(ascending=False)
        print("\nTop 10 Short Tank Specialist Features (Stage 4):")
        for i, (feature, importance) in enumerate(stage4_importance.head(10).items()):
            is_seasonal = any(season_word in feature.lower() for season_word in ['month', 'season', 'quarter', 'construction', 'capacity', 'sin', 'cos'])
            marker = "[SEASONAL]" if is_seasonal else ""
            print(f"{i+1:2d}. {feature:<35} {importance:.4f} {marker}")

    # Count seasonal features in top 10
    seasonal_count_base = sum(1 for feature in base_avg_importance.head(10).index
                             if any(season_word in feature.lower() for season_word in ['month', 'season', 'quarter', 'construction', 'capacity', 'sin', 'cos']))

    print(f"\nSeasonality Impact:")
    print(f"Seasonal features in top 10 base features: {seasonal_count_base}/10")

    return importance_df

def evaluate_performance(test_data, predictions):
    """Evaluate model performance"""
    y_true = test_data['TotalCost'].values

    overall_mae = mean_absolute_error(y_true, predictions)
    overall_rmse = np.sqrt(mean_squared_error(y_true, predictions))
    overall_r2 = r2_score(y_true, predictions)

    # By tank type
    short_mask = test_data['HEIGHT'] < 50.0
    normal_mask = test_data['HEIGHT'] >= 50.0

    results = {
        'overall_mae': overall_mae,
        'overall_rmse': overall_rmse,
        'overall_r2': overall_r2,
        'total_samples': len(test_data)
    }

    if short_mask.sum() > 0:
        short_mae = mean_absolute_error(y_true[short_mask], predictions[short_mask])
        short_rmse = np.sqrt(mean_squared_error(y_true[short_mask], predictions[short_mask]))
        short_r2 = r2_score(y_true[short_mask], predictions[short_mask])

        results.update({
            'short_tank_mae': short_mae,
            'short_tank_rmse': short_rmse,
            'short_tank_r2': short_r2,
            'short_tank_samples': short_mask.sum()
        })

    if normal_mask.sum() > 0:
        normal_mae = mean_absolute_error(y_true[normal_mask], predictions[normal_mask])
        normal_rmse = np.sqrt(mean_squared_error(y_true[normal_mask], predictions[normal_mask]))
        normal_r2 = r2_score(y_true[normal_mask], predictions[normal_mask])

        results.update({
            'normal_tank_mae': normal_mae,
            'normal_tank_rmse': normal_rmse,
            'normal_tank_r2': normal_r2,
            'normal_tank_samples': normal_mask.sum()
        })

    return results

def create_detailed_error_analysis(test_data, predictions):
    """Create detailed error analysis Excel file"""
    print("\nCreating detailed error analysis...")
    
    # Calculate errors
    y_true = test_data['TotalCost'].values
    errors = predictions - y_true
    abs_errors = np.abs(errors)
    percentage_errors = (errors / y_true) * 100
    abs_percentage_errors = np.abs(percentage_errors)
    
    # Create detailed analysis dataframe
    error_analysis = test_data.copy()
    error_analysis['Predicted_Cost'] = predictions
    error_analysis['Actual_Cost'] = y_true
    error_analysis['Error'] = errors
    error_analysis['Absolute_Error'] = abs_errors
    error_analysis['Percentage_Error'] = percentage_errors
    error_analysis['Absolute_Percentage_Error'] = abs_percentage_errors
    
    # Add tank type classification
    error_analysis['Tank_Type'] = np.where(error_analysis['HEIGHT'] < 50.0, 'Short Tank', 'Normal Tank')
    
    # Add error categories
    error_analysis['Error_Category'] = pd.cut(
        error_analysis['Absolute_Percentage_Error'],
        bins=[0, 5, 10, 20, 50, float('inf')],
        labels=['Excellent (<5%)', 'Good (5-10%)', 'Fair (10-20%)', 'Poor (20-50%)', 'Very Poor (>50%)']
    )
    
    # Add cost categories
    error_analysis['Cost_Category'] = pd.cut(
        error_analysis['Actual_Cost'],
        bins=[0, 20000, 50000, 100000, 200000, float('inf')],
        labels=['Low (<$20K)', 'Medium ($20K-$50K)', 'High ($50K-$100K)', 'Very High ($100K-$200K)', 'Premium (>$200K)']
    )
    
    # Select key columns for the output
    output_columns = [
        'ORDER', 'HEIGHT', 'DIAMETER', 'WEIGHT', 'PRESSURE', 'VACUUM',
        'MATERIAL', 'BOTTOM', 'PLANT', 'ORDERCLASS', 'Tank_Type',
        'Actual_Cost', 'Predicted_Cost', 'Error', 'Absolute_Error',
        'Percentage_Error', 'Absolute_Percentage_Error',
        'Error_Category', 'Cost_Category'
    ]
    
    # Add project ID and parent project columns if available
    if 'Project ID' in error_analysis.columns:
        output_columns.insert(1, 'Project ID')
    if 'Parent Project' in error_analysis.columns:
        output_columns.insert(2, 'Parent Project')
    if 'ProjectID' in error_analysis.columns:
        output_columns.insert(1, 'ProjectID')
    if 'ParentProject' in error_analysis.columns:
        output_columns.insert(2, 'ParentProject')
    
    # Add date columns if available
    if 'Created Date' in error_analysis.columns:
        output_columns.insert(-2, 'Created Date')
    if 'Actual Ship Date' in error_analysis.columns:
        output_columns.insert(-2, 'Actual Ship Date')
    
    # Filter to available columns
    available_columns = [col for col in output_columns if col in error_analysis.columns]
    detailed_errors = error_analysis[available_columns].copy()
    
    # Sort by absolute error (highest first)
    detailed_errors = detailed_errors.sort_values('Absolute_Error', ascending=False)
    
    # Create summary statistics
    summary_stats = pd.DataFrame({
        'Metric': ['Total Samples', 'Overall MAE', 'Overall RMSE', 'Overall R²',
                   'Short Tank MAE', 'Short Tank RMSE', 'Short Tank R²', 'Short Tank Samples',
                   'Normal Tank MAE', 'Normal Tank RMSE', 'Normal Tank R²', 'Normal Tank Samples',
                   'Excellent Predictions (<5% error)', 'Good Predictions (5-10% error)',
                   'Fair Predictions (10-20% error)', 'Poor Predictions (20-50% error)',
                   'Very Poor Predictions (>50% error)'],
        'Value': [
            len(test_data),
            mean_absolute_error(y_true, predictions),
            np.sqrt(mean_squared_error(y_true, predictions)),
            r2_score(y_true, predictions),
            mean_absolute_error(y_true[error_analysis['Tank_Type'] == 'Short Tank'], 
                              predictions[error_analysis['Tank_Type'] == 'Short Tank']) if (error_analysis['Tank_Type'] == 'Short Tank').any() else 'N/A',
            np.sqrt(mean_squared_error(y_true[error_analysis['Tank_Type'] == 'Short Tank'], 
                                     predictions[error_analysis['Tank_Type'] == 'Short Tank'])) if (error_analysis['Tank_Type'] == 'Short Tank').any() else 'N/A',
            r2_score(y_true[error_analysis['Tank_Type'] == 'Short Tank'], 
                    predictions[error_analysis['Tank_Type'] == 'Short Tank']) if (error_analysis['Tank_Type'] == 'Short Tank').any() else 'N/A',
            (error_analysis['Tank_Type'] == 'Short Tank').sum(),
            mean_absolute_error(y_true[error_analysis['Tank_Type'] == 'Normal Tank'], 
                              predictions[error_analysis['Tank_Type'] == 'Normal Tank']) if (error_analysis['Tank_Type'] == 'Normal Tank').any() else 'N/A',
            np.sqrt(mean_squared_error(y_true[error_analysis['Tank_Type'] == 'Normal Tank'], 
                                     predictions[error_analysis['Tank_Type'] == 'Normal Tank'])) if (error_analysis['Tank_Type'] == 'Normal Tank').any() else 'N/A',
            r2_score(y_true[error_analysis['Tank_Type'] == 'Normal Tank'], 
                    predictions[error_analysis['Tank_Type'] == 'Normal Tank']) if (error_analysis['Tank_Type'] == 'Normal Tank').any() else 'N/A',
            (error_analysis['Tank_Type'] == 'Normal Tank').sum(),
            (error_analysis['Error_Category'] == 'Excellent (<5%)').sum(),
            (error_analysis['Error_Category'] == 'Good (5-10%)').sum(),
            (error_analysis['Error_Category'] == 'Fair (10-20%)').sum(),
            (error_analysis['Error_Category'] == 'Poor (20-50%)').sum(),
            (error_analysis['Error_Category'] == 'Very Poor (>50%)').sum()
        ]
    })
    
    # Write to Excel with multiple sheets
    filename = 'test_errors_detailed_seasonality.xlsx'
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        # Main detailed errors sheet
        detailed_errors.to_excel(writer, sheet_name='Detailed_Errors', index=False)
        
        # Summary statistics sheet
        summary_stats.to_excel(writer, sheet_name='Summary_Stats', index=False)
        
        # Error category breakdown
        error_breakdown = error_analysis.groupby(['Tank_Type', 'Error_Category']).size().unstack(fill_value=0)
        error_breakdown.to_excel(writer, sheet_name='Error_Breakdown')
        
        # Cost category analysis
        cost_analysis = error_analysis.groupby('Cost_Category').agg({
            'Absolute_Error': ['mean', 'median', 'std', 'count'],
            'Absolute_Percentage_Error': ['mean', 'median', 'std']
        }).round(2)
        cost_analysis.to_excel(writer, sheet_name='Cost_Analysis')
    
    print(f"Detailed error analysis saved to: {filename}")
    return detailed_errors, summary_stats

def main():
    """Main function with seasonality"""
    print("4-STAGE LIST PRICE MODEL WITH SEASONALITY")
    print("=" * 41)

    # Load data
    train_data, test_data = load_data()

    # Train models
    models, base_encoders, short_encoders, base_features, short_features = train_4stage_model_with_seasonality(train_data)

    # Make predictions
    predictions = predict_4stage_with_seasonality(models, test_data, base_encoders, short_encoders, base_features, short_features)

    # Get feature importance
    importance_df = get_feature_importance(models, base_features, short_features)

    # Evaluate
    results = evaluate_performance(test_data, predictions)
    
    # Create detailed error analysis
    detailed_errors, summary_stats = create_detailed_error_analysis(test_data, predictions)

    # Display results
    print("\n4-STAGE MODEL WITH SEASONALITY RESULTS:")
    print("=" * 37)
    print(f"Total test samples: {len(test_data)}")
    print(f"Overall MAE: ${results['overall_mae']:,.0f}")
    print(f"Overall RMSE: ${results['overall_rmse']:,.0f}")
    print(f"Overall R2: {results['overall_r2']:.4f}")

    if 'short_tank_mae' in results:
        print(f"\nShort Tank Performance:")
        print(f"  Samples: {results['short_tank_samples']}")
        print(f"  MAE: ${results['short_tank_mae']:,.0f}")
        print(f"  RMSE: ${results['short_tank_rmse']:,.0f}")
        print(f"  R2: {results['short_tank_r2']:.4f}")

    if 'normal_tank_mae' in results:
        print(f"\nNormal Tank Performance:")
        print(f"  Samples: {results['normal_tank_samples']}")
        print(f"  MAE: ${results['normal_tank_mae']:,.0f}")
        print(f"  RMSE: ${results['normal_tank_rmse']:,.0f}")
        print(f"  R2: {results['normal_tank_r2']:.4f}")

    # Compare with baselines
    corrected_baseline = 9614
    list_price_baseline = 9645
    original_4stage = 9910

    improvement_corrected = ((corrected_baseline - results['overall_mae']) / corrected_baseline) * 100
    improvement_list_price = ((list_price_baseline - results['overall_mae']) / list_price_baseline) * 100
    improvement_4stage = ((original_4stage - results['overall_mae']) / original_4stage) * 100

    print(f"\nCOMPARISON WITH BASELINES:")
    print(f"Corrected System: ${corrected_baseline:,}")
    print(f"List Price Model: ${list_price_baseline:,}")
    print(f"4-Stage Original: ${original_4stage:,}")
    print(f"4-Stage + Seasonality: ${results['overall_mae']:,.0f}")
    print(f"vs Corrected: {improvement_corrected:+.2f}%")
    print(f"vs List Price: {improvement_list_price:+.2f}%")
    print(f"vs 4-Stage Original: {improvement_4stage:+.2f}%")

    success_corrected = results['overall_mae'] < corrected_baseline
    success_list_price = results['overall_mae'] < list_price_baseline
    success_4stage = results['overall_mae'] < original_4stage

    print(f"\nSUCCESS:")
    print(f"Better than Corrected: {'YES' if success_corrected else 'NO'}")
    print(f"Better than List Price: {'YES' if success_list_price else 'NO'}")
    print(f"Better than 4-Stage Original: {'YES' if success_4stage else 'NO'}")

    return results, importance_df, detailed_errors, summary_stats

if __name__ == "__main__":
    results, importance_df, detailed_errors, summary_stats = main()