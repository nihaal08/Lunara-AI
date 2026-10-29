import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
from imblearn.over_sampling import BorderlineSMOTE
import joblib
import warnings
warnings.filterwarnings('ignore')

# Section 1: Loading and Cleaning Data
# Loading the maternal health dataset
df = pd.read_csv('Maternal Health Risk Data Set.csv')

# Removing less relevant column
columns_to_drop = ['BodyTemp']
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

# Dropping rows with invalid target values (if any)
valid_risks = ['low risk', 'mid risk', 'high risk']
df = df[df['RiskLevel'].isin(valid_risks)].copy()

# Section 2: Preprocessing Numeric Columns
# Defining numeric columns
numeric_columns = ['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'HeartRate']

# Converting to numeric and handling outliers
num_imputer = SimpleImputer(strategy='median')
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    # Capping outliers using IQR
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    df[col] = df[col].clip(lower=Q1 - 1.25 * IQR, upper=Q3 + 1.25 * IQR)
df[numeric_columns] = num_imputer.fit_transform(df[numeric_columns])

# Section 3: Feature Engineering
# Creating BP ratio
df['BP_ratio'] = df['SystolicBP'] / df['DiastolicBP'].replace(0, 1)  # Avoid division by zero

# Creating Age categories
df['Age_category'] = pd.cut(df['Age'], 
                           bins=[0, 18, 35, 100], 
                           labels=[0, 1, 2], 
                           include_lowest=True)
df['Age_category'] = df['Age_category'].cat.codes
df['Age_category'] = df['Age_category'].replace(-1, 1)

# Creating high BS indicator
df['High_BS'] = (df['BS'] > 11).astype(int)

# Adding interaction terms
df['SystolicBP_BS_interaction'] = df['SystolicBP'] * df['BS']
df['Age_BS_interaction'] = df['Age'] * df['BS']

# Checking for NaNs
if df.isna().any().any():
    print("NaNs found in columns:", df.columns[df.isna().any()].tolist())
    df = df.fillna(df.median(numeric_only=True))

# Section 4: Preparing Features and Target
# Encoding target
label_encoder = LabelEncoder()
df['RiskLevel'] = label_encoder.fit_transform(df['RiskLevel'])

# Splitting features and target
X = df.drop('RiskLevel', axis=1)
y = df['RiskLevel']

# Standardizing features
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Final NaN check
if X.isna().any().any():
    print("NaNs in X after preprocessing:", X.columns[X.isna().any()].tolist())

# Section 5: Training the Model
# Applying Borderline-SMOTE for multiclass imbalance
smote = BorderlineSMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, 
                                                    test_size=0.2, random_state=42)

# Initializing XGBoost Classifier
xgb_model = XGBClassifier(random_state=42, eval_metric='mlogloss')

# Expanded hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}
grid_search = GridSearchCV(xgb_model, param_grid, cv=10, scoring='f1_weighted', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)

# Cross-validation score
cv_scores = cross_val_score(best_model, X_train, y_train, cv=10, scoring='accuracy')
print("Cross-Validation Accuracy: {:.2f} ± {:.2f}".format(cv_scores.mean(), cv_scores.std()))

# Evaluating on test set
y_pred = best_model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Section 6: Saving the Model and Artifacts
joblib.dump(best_model, 'maternal_risk_model.pkl')
joblib.dump(label_encoder, 'label_encoder_maternal.pkl')
joblib.dump(scaler, 'scaler_maternal.pkl')
joblib.dump(X_train.columns, 'columns_maternal.pkl')
print("Model and artifacts saved successfully.")

# Section 7: Function for Loading and Predicting
def load_and_predict(new_data):
    model = joblib.load('maternal_risk_model.pkl')
    label_encoder = joblib.load('label_encoder_maternal.pkl')
    scaler = joblib.load('scaler_maternal.pkl')
    saved_columns = joblib.load('columns_maternal.pkl')

    # Feature engineering for new data
    new_data['BP_ratio'] = new_data['SystolicBP'] / new_data['DiastolicBP'].replace(0, 1)
    new_data['Age_category'] = pd.cut(new_data['Age'], 
                                     bins=[0, 18, 35, 100], 
                                     labels=[0, 1, 2], 
                                     include_lowest=True)
    new_data['Age_category'] = new_data['Age_category'].cat.codes
    new_data['Age_category'] = new_data['Age_category'].replace(-1, 1)
    new_data['High_BS'] = (new_data['BS'] > 11).astype(int)
    new_data['SystolicBP_BS_interaction'] = new_data['SystolicBP'] * new_data['BS']
    new_data['Age_BS_interaction'] = new_data['Age'] * new_data['BS']

    # Ensuring same columns
    new_data = new_data.reindex(columns=saved_columns, fill_value=0)

    # Scaling features
    new_data = scaler.transform(new_data)

    # Making predictions
    predictions = model.predict(new_data)
    predicted_labels = label_encoder.inverse_transform(predictions)

    return predicted_labels

# Section 8: Example Prediction
new_data = pd.DataFrame({
    'Age': [30], 'SystolicBP': [120], 'DiastolicBP': [80], 'BS': [7.0], 'HeartRate': [70]
})

predictions = load_and_predict(new_data)
print("\nPrediction for new data:", predictions)