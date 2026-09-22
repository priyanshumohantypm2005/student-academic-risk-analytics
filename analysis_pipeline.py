"""
AI-Powered Student Academic Performance & Risk Analytics Platform
Core Data Analytics, Machine Learning & Feature Engineering Pipeline
Ground-truth analytics based strictly on uploaded student dataset (1,194 records)
"""

import json
import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)

def clean_attendance(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    if '-' in s:
        parts = s.split('-')
        try:
            return (float(parts[0]) + float(parts[1])) / 2.0
        except:
            return np.nan
    try:
        return float(s)
    except:
        return np.nan

def clean_area(val):
    if pd.isna(val):
        return 'Other'
    s = str(val).strip()
    # Normalize known variations
    s_low = s.lower()
    if 'schince' in s_low or 'data science' in s_low:
        return 'Data Science'
    elif 'network' in s_low:
        return 'Networking'
    elif 'program' in s_low:
        return 'Programming'
    elif 'web develop' in s_low or 'web developing' in s_low:
        return 'Web Development'
    elif 'hardware' in s_low:
        return 'Hardware'
    elif 'software' in s_low:
        return 'Software'
    elif 'artificial intelligence' in s_low or 'ai' == s_low:
        return 'Artificial Intelligence'
    elif 'machine learning' in s_low or 'ml' == s_low:
        return 'Machine Learning'
    elif 'cyber' in s_low or 'syber' in s_low:
        return 'Cyber Security'
    elif 'ui/ux' in s_low or 'ui' in s_low:
        return 'UI/UX Design'
    elif 'management' in s_low:
        return 'Management'
    elif 'game' in s_low:
        return 'Game Development'
    return s.title()

def clean_skills(val):
    if pd.isna(val):
        return 'None / Not Specified'
    s = str(val).strip()
    s_low = s.lower()
    if 'none' in s_low or 'not' in s_low or "don't" in s_low or 'nothing' in s_low or 'no skill' in s_low or "haven't" in s_low:
        return 'None / Developing'
    if 'programming' in s_low:
        return 'Programming'
    if 'web develop' in s_low:
        return 'Web Development'
    if 'networking' in s_low:
        return 'Networking'
    if 'cyber' in s_low:
        return 'Cyber Security'
    if 'software development' in s_low:
        return 'Software Development'
    if 'machine learning' in s_low:
        return 'Machine Learning'
    if 'artificial intelligence' in s_low:
        return 'Artificial Intelligence'
    if 'graphics' in s_low or 'graphic' in s_low:
        return 'Graphics Design'
    return s.title()

def load_and_preprocess_data(csv_path='student_data.csv'):
    df = pd.read_csv(csv_path)
    
    # Generate clean Student ID
    df['student_id'] = [f'STU{1000 + i}' for i in range(len(df))]
    
    # 1. Clean attendance
    df['attendance_clean'] = df['Average attendance on class'].apply(clean_attendance)
    # If any attendance missing, impute with median attendance
    df['attendance_clean'] = df['attendance_clean'].fillna(df['attendance_clean'].median())
    
    # 2. Clean Health issues
    df['health_issues_clean'] = df['Do you have any health issues?'].astype(str).str.strip().str.capitalize()
    df['health_issues_clean'] = df['health_issues_clean'].replace({'N': 'No', 'No': 'No', 'Yes': 'Yes'})
    
    # 3. Clean Relationship Status
    df['relationship_clean'] = df['What is your relationship status?'].astype(str).str.strip()
    df['relationship_clean'] = df['relationship_clean'].replace({'In a relationship': 'Relationship'})
    
    # 4. Clean Interested Area
    df['interested_area_clean'] = df['What is you interested area?'].apply(clean_area)
    
    # 5. Clean Skills
    df['skills_clean'] = df['What are the skills do you have ?'].apply(clean_skills)
    
    # 6. Target definition based on median CGPA
    median_cgpa = float(df['What is your current CGPA?'].median())
    # Risk = 1 if below median, 0 if Good (>= median)
    df['performance_risk_label'] = np.where(df['What is your current CGPA?'] < median_cgpa, 'At Risk', 'Good')
    df['performance_risk_binary'] = np.where(df['What is your current CGPA?'] < median_cgpa, 1, 0)
    
    return df, median_cgpa

def run_ml_pipeline(df):
    # Selected features (strict target leakage prevention: EXCLUDE 'What is your current CGPA?')
    # Also exclude student_id and raw text variants
    
    numerical_features = [
        'Age',
        'How many hour do you study daily?',
        'How many times do you seat for study in a day?',
        'How many hour do you spent daily in social media?',
        'attendance_clean',
        'How many hour do you spent daily on your skill development?',
        'What was your previous SGPA?',
        'How many Credit did you have completed?',
        'What is your monthly family income?',
        'Current Semester'
    ]
    
    categorical_features = [
        'Gender',
        'Do you have meritorious scholarship ?',
        'Do you use University transportation?',
        'What is your preferable learning mode?',
        'Do you use smart phone?',
        'Do you have personal Computer?',
        'Status of your English language proficiency',
        'Did you ever fall in probation?',
        'Did you ever got suspension?',
        'Do you attend in teacher consultancy for any kind of academical problems?',
        'skills_clean',
        'interested_area_clean',
        'relationship_clean',
        'Are you engaged with any co-curriculum activities?',
        'With whom you are living with?',
        'health_issues_clean',
        'Do you have any physical disabilities?'
    ]
    
    X = df[numerical_features + categorical_features]
    y = df['performance_risk_binary']
    
    # Train-test split (80-20, stratified, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transformer, numerical_features),
        ('cat', cat_transformer, categorical_features)
    ])
    
    # 1. Logistic Regression Pipeline
    lr_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
    
    lr_pipeline.fit(X_train, y_train)
    y_pred_lr = lr_pipeline.predict(X_test)
    y_proba_lr = lr_pipeline.predict_proba(X_test)[:, 1]
    
    # 2. Random Forest Pipeline
    rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42))
    ])
    
    rf_pipeline.fit(X_train, y_train)
    y_pred_rf = rf_pipeline.predict(X_test)
    y_proba_rf = rf_pipeline.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    def get_metrics(y_true, y_pred, y_proba):
        acc = float(accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        roc_auc = float(roc_auc_score(y_true, y_proba))
        cm = confusion_matrix(y_true, y_pred).tolist()
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        return {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist()
        }
    
    metrics_lr = get_metrics(y_test, y_pred_lr, y_proba_lr)
    metrics_rf = get_metrics(y_test, y_pred_rf, y_proba_rf)
    
    # Feature Importance for Random Forest
    rf_classifier = rf_pipeline.named_steps['classifier']
    preprocessor_fitted = rf_pipeline.named_steps['preprocessor']
    
    # Get feature names from onehot
    cat_encoder = preprocessor_fitted.named_transformers_['cat'].named_steps['onehot']
    encoded_cat_names = list(cat_encoder.get_feature_names_out(categorical_features))
    all_feature_names = numerical_features + encoded_cat_names
    
    importances = rf_classifier.feature_importances_
    feat_imp = pd.DataFrame({
        'feature': all_feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # Predictions for all students in dataset
    all_pred_rf = rf_pipeline.predict(X)
    all_proba_rf = rf_pipeline.predict_proba(X)[:, 1]  # proba of At Risk
    
    df['predicted_risk_binary'] = all_pred_rf
    df['predicted_risk_label'] = np.where(all_pred_rf == 1, 'At Risk', 'Good')
    df['predicted_risk_probability'] = all_proba_rf
    df['predicted_confidence'] = np.where(all_pred_rf == 1, all_proba_rf, 1 - all_proba_rf)
    
    return {
        'metrics_lr': metrics_lr,
        'metrics_rf': metrics_rf,
        'top_features': feat_imp.head(25).to_dict(orient='records'),
        'df_processed': df,
        'numerical_features': numerical_features,
        'categorical_features': categorical_features
    }

if __name__ == '__main__':
    print('Executing Data Cleaning and ML Pipeline...')
    df, median_cgpa = load_and_preprocess_data('student_data.csv')
    print(f'Data loaded: {len(df)} records. Median CGPA = {median_cgpa:.4f}')
    
    results = run_ml_pipeline(df)
    
    print('\n--- Logistic Regression Metrics ---')
    for k, v in results['metrics_lr'].items():
        if k not in ['fpr', 'tpr']:
            print(f'  {k}: {v}')
            
    print('\n--- Random Forest Metrics ---')
    for k, v in results['metrics_rf'].items():
        if k not in ['fpr', 'tpr']:
            print(f'  {k}: {v}')
            
    print('\n--- Top 10 Features (Random Forest) ---')
    for f in results['top_features'][:10]:
        print(f"  {f['feature']}: {f['importance']:.4f}")
        
    # Save processed dataset
    df_out = results['df_processed']
    df_out.to_csv('student_data_processed.csv', index=False)
    print('\nSaved student_data_processed.csv')
    
    # Save metrics JSON for Streamlit and report
    summary_metrics = {
        'dataset_summary': {
            'total_students': len(df_out),
            'total_columns': 31,
            'median_cgpa': median_cgpa,
            'mean_cgpa': float(df_out['What is your current CGPA?'].mean()),
            'min_cgpa': float(df_out['What is your current CGPA?'].min()),
            'max_cgpa': float(df_out['What is your current CGPA?'].max()),
            'at_risk_count': int((df_out['performance_risk_label'] == 'At Risk').sum()),
            'at_risk_pct': float((df_out['performance_risk_label'] == 'At Risk').mean() * 100),
            'good_count': int((df_out['performance_risk_label'] == 'Good').sum()),
            'good_pct': float((df_out['performance_risk_label'] == 'Good').mean() * 100),
            'avg_attendance': float(df_out['attendance_clean'].mean()),
            'avg_study_hours': float(df_out['How many hour do you study daily?'].mean()),
            'avg_social_hours': float(df_out['How many hour do you spent daily in social media?'].mean()),
            'probation_count': int((df_out['Did you ever fall in probation?'] == 'Yes').sum()),
            'probation_pct': float((df_out['Did you ever fall in probation?'] == 'Yes').mean() * 100),
            'scholarship_count': int((df_out['Do you have meritorious scholarship ?'] == 'Yes').sum()),
            'scholarship_pct': float((df_out['Do you have meritorious scholarship ?'] == 'Yes').mean() * 100),
        },
        'models': {
            'logistic_regression': {k: v for k, v in results['metrics_lr'].items() if k not in ['fpr', 'tpr']},
            'random_forest': {k: v for k, v in results['metrics_rf'].items() if k not in ['fpr', 'tpr']}
        },
        'roc_curves': {
            'logistic_regression': {'fpr': results['metrics_lr']['fpr'], 'tpr': results['metrics_lr']['tpr']},
            'random_forest': {'fpr': results['metrics_rf']['fpr'], 'tpr': results['metrics_rf']['tpr']}
        },
        'top_features': results['top_features']
    }
    
    with open('pipeline_results.json', 'w') as f:
        json.dump(summary_metrics, f, indent=2)
    print('Saved pipeline_results.json successfully!')
