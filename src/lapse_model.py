import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, confusion_matrix

def train_models():
    os.makedirs("outputs", exist_ok=True)
    df = pd.read_csv("outputs/part_b/lapse_dataset.csv")
    
    X = df.drop(columns=['customer_id', 'lapsed'])
    y = df['lapsed']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    numeric_features = ['recency_to_cutoff', 'frequency', 'monetary', 'avg_discount', 'avg_delivery', 'cancellation_rate', 'age', 'tenure_days']
    categorical_features = ['city', 'gender', 'favourite_category']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
        ]
    )
    
    # Random Forest Model
    rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    param_grid_rf = {
        'classifier__max_depth': [3, 5, 8],
        'classifier__min_samples_leaf': [5, 10, 20]
    }
    
    grid_rf = GridSearchCV(rf_pipeline, param_grid_rf, cv=5, scoring='f1', n_jobs=-1)
    grid_rf.fit(X_train, y_train)
    
    print(f"Best RF Parameters: {grid_rf.best_params_}")
    print(f"Best RF CV F1: {grid_rf.best_score_:.3f}")
    
    best_model = grid_rf.best_estimator_
    y_pred = best_model.predict(X_test)
    test_f1 = f1_score(y_test, y_pred, zero_division=0)
    print(f"Test F1 (Lapsed class): {test_f1:.3f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    
    # Save model
    joblib.dump(best_model, "outputs/lapse_model.joblib")
    print("Model saved to outputs/lapse_model.joblib successfully.")
    
    # Generate Retention List for Premium customers
    segments = pd.read_csv("outputs/customer_segments.csv")
    premium_cids = segments[segments['is_premium'] == True]['customer_id'].tolist()
    
    test_indices = X_test.index
    test_full = df.loc[test_indices].copy()
    test_full['lapse_probability'] = best_model.predict_proba(X_test)[:, 1]
    
    retention_list = test_full[test_full['customer_id'].isin(premium_cids)].sort_values(by='lapse_probability', ascending=False)
    retention_out = retention_list[['customer_id', 'lapse_probability', 'lapsed']].rename(columns={'lapsed': 'actually_lapsed'})
    retention_out.to_csv("outputs/retention_list.csv", index=False)
    print(f"Retention list saved with {len(retention_out)} records.")

if __name__ == "__main__":
    train_models()