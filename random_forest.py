import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler



train_df = pd.read_csv('train.csv')
val_df = pd.read_csv('val.csv')
test_df = pd.read_csv('test.csv')

print(f"Train shape: {train_df.shape}")
print(f"Val shape:   {val_df.shape}")
print(f"Test shape:  {test_df.shape}")


label_column = 'Label'

# Create binary target: 1 = attack/anomaly, 0 = normal/benign
train_df['target'] = (train_df[label_column] != 'Benign').astype(int)
val_df['target'] = (val_df[label_column] != 'Benign').astype(int)
test_df['target'] = (test_df[label_column] != 'Benign').astype(int)

# Drop the original label column
train_df = train_df.drop(label_column, axis=1)
val_df = val_df.drop(label_column, axis=1)
test_df = test_df.drop(label_column, axis=1)


X_train = train_df.drop('target', axis=1)
y_train = train_df['target']

X_val = val_df.drop('target', axis=1)


X_test = test_df.drop('target', axis=1)

print("\nPreprocessing data...")

# Handle infinite values and NaN
X_train = X_train.replace([np.inf, -np.inf], np.nan)
X_val = X_val.replace([np.inf, -np.inf], np.nan)
X_test = X_test.replace([np.inf, -np.inf], np.nan)

# Fill NaN with median from training data
X_train = X_train.fillna(X_train.median())
X_val = X_val.fillna(X_train.median())
X_test = X_test.fillna(X_train.median())

# Encode categorical features
categorical_cols = X_train.select_dtypes(include=['object']).columns
if len(categorical_cols) > 0:
    print(f"Encoding {len(categorical_cols)} categorical columns...")
    for col in categorical_cols:
        le = LabelEncoder()
        # Fit on combined data to avoid unseen categories
        combined = pd.concat([X_train[col], X_val[col], X_test[col]], axis=0).astype(str)
        le.fit(combined)
        X_train[col] = le.transform(X_train[col].astype(str))
        X_val[col] = le.transform(X_val[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train_scaled, y_train)

print("\nMaking predictions on test set...")

# Get probability for class 1 (attack) - this is the confidence
y_proba = rf.predict_proba(X_test_scaled)[:, 1]

# Get binary classification result (threshold = 0.5)
y_pred = (y_proba >= 0.5).astype(int)


results_df = pd.DataFrame({
    'prediction': y_pred,      # 0 or 1
    'confidence': y_proba      # number from 0 to 1
})

# Save to CSV
output_file = 'predictions.csv'
results_df.to_csv(output_file, index=False)

print(f"\n Results saved to: {output_file}")
print(f"   Total predictions: {len(results_df)}")
print(f"\nSample of results (first 10 rows):")
print(results_df.head(10).to_string())

# Optional: Show distribution of predictions
print(f"\nPrediction distribution:")
print(f"  Class 0 (normal): {(y_pred == 0).sum()} samples ({((y_pred == 0).sum()/len(y_pred)*100):.1f}%)")
print(f"  Class 1 (attack): {(y_pred == 1).sum()} samples ({((y_pred == 1).sum()/len(y_pred)*100):.1f}%)")
