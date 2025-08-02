import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings('ignore')

# Section 1: Loading and Cleaning Data
# Loading the gestational diabetes dataset
try:
    df = pd.read_csv('Gestational_Diabetic.csv')
except FileNotFoundError:
    print("Error: 'Gestational_Diabetic.csv' not found. Please ensure the file is in the working directory.")
    exit(1)

# Debugging: Check initial DataFrame
print("Initial DataFrame shape:", df.shape)
print("Columns in DataFrame:", df.columns.tolist())

# Removing less relevant columns
columns_to_drop = ['Case Number', 'Gestation in previous Pregnancy', 'unexplained prenetal loss', 'Hemoglobin']
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

# Renaming columns for consistency
df = df.rename(columns={'Sys BP': 'Sys_BP', 'Dia BP': 'Dia_BP', 'Class Label(GDM /Non GDM)': 'Class_Label'})

# Debugging: Check Class_Label values
print("Unique Class_Label values before cleaning:", df['Class_Label'].unique())

# Cleaning Class_Label: Strip whitespace and handle numeric values
df['Class_Label'] = df['Class_Label'].astype(str).str.strip()
# Convert numeric labels to string if present (e.g., '0' -> 'GDM', '1' -> 'Non GDM')
df['Class_Label'] = df['Class_Label'].replace({'0': 'GDM', '1': 'Non GDM'})

# Filtering rows with valid target values
valid_labels = ['GDM', 'Non GDM']
df = df[df['Class_Label'].isin(valid_labels)].copy()

# Debugging: Check DataFrame after filtering
print("DataFrame shape after filtering:", df.shape)
if df.empty:
    print("Error: DataFrame is empty after filtering. Possible issues with Class_Label values.")
    print("Unique Class_Label values:", df['Class_Label'].unique())
    exit(1)

# Section 2: Preprocessing Numeric and Categorical Columns
# Defining numeric and categorical columns
numeric_columns = ['Age', 'No of Pregnancy', 'BMI', 'HDL', 'Sys_BP', 'Dia_BP', 'OGTT']
categorical_columns = ['Family History', 'Large Child or Birth Default', 'PCOS', 'Sedentary Lifestyle', 'Prediabetes']

# Validating columns
missing_numeric = [col for col in numeric_columns if col not in df.columns]
if missing_numeric:
    print(f"Error: Missing numeric columns: {missing_numeric}")
    exit(1)

# Converting to numeric and handling outliers
num_imputer = SimpleImputer(strategy='median')
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    # Capping outliers using IQR
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    df[col] = df[col].clip(lower=Q1 - 1.25 * IQR, upper=Q3 + 1.25 * IQR)

# Check if numeric columns have data before imputation
if df[numeric_columns].empty:
    print("Error: Numeric columns are empty before imputation.")
    print("Numeric columns:", numeric_columns)
    exit(1)

df[numeric_columns] = num_imputer.fit_transform(df[numeric_columns])

# Imputing categorical columns
cat_imputer = SimpleImputer(strategy='most_frequent')
df[categorical_columns] = cat_imputer.fit_transform(df[categorical_columns])
for col in categorical_columns:
    df[col] = df[col].astype(int)

# Section 3: Feature Engineering
# Creating BP ratio
df['BP_ratio'] = df['Sys_BP'] / df['Dia_BP'].replace(0, 1)  # Avoid division by zero

# Creating Age categories
df['Age_category'] = pd.cut(df['Age'], 
                           bins=[0, 25, 35, 100], 
                           labels=[0, 1, 2], 
                           include_lowest=True)
df['Age_category'] = df['Age_category'].cat.codes
df['Age_category'] = df['Age_category'].replace(-1, 1)

# Creating high OGTT indicator
df['High_OGTT'] = (df['OGTT'] > 140).astype(int)

# Creating BMI categories
df['BMI_category'] = pd.cut(df['BMI'], 
                           bins=[0, 25, 30, 100], 
                           labels=[0, 1, 2], 
                           include_lowest=True)
df['BMI_category'] = df['BMI_category'].cat.codes
df['BMI_category'] = df['BMI_category'].replace(-1, 1)

# Adding interaction terms
df['OGTT_BMI_interaction'] = df['OGTT'] * df['BMI']
df['Age_OGTT_interaction'] = df['Age'] * df['OGTT']

# Checking for NaNs
if df.isna().any().any():
    print("NaNs found in columns:", df.columns[df.isna().any()].tolist())
    df = df.fillna(df.median(numeric_only=True))

# Section 4: Preparing Features and Target
# Encoding target
label_encoder = LabelEncoder()
df['Class_Label'] = label_encoder.fit_transform(df['Class_Label'])  # GDM=0, Non GDM=1

# Splitting features and target
X = df.drop('Class_Label', axis=1)
y = df['Class_Label']

# Standardizing numeric features
scaler = StandardScaler()
X[numeric_columns + ['BP_ratio', 'OGTT_BMI_interaction', 'Age_OGTT_interaction']] = scaler.fit_transform(
    X[numeric_columns + ['BP_ratio', 'OGTT_BMI_interaction', 'Age_OGTT_interaction']])

# Final NaN check
if X.isna().any().any():
    print("NaNs in X after preprocessing:", X.columns[X.isna().any()].tolist())

# Section 5: Training the Model
# Applying SMOTE for class imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, 
                                                    test_size=0.2, random_state=42)

# Initializing XGBoost Classifier
xgb_model = XGBClassifier(random_state=42, eval_metric='logloss')

# Expanded hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}
grid_search = GridSearchCV(xgb_model, param_grid, cv=10, scoring='f1', n_jobs=-1)
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
joblib.dump(best_model, 'gdm_model.pkl')
joblib.dump(label_encoder, 'label_encoder_gdm.pkl')
joblib.dump(scaler, 'scaler_gdm.pkl')
joblib.dump(X_train.columns, 'columns_gdm.pkl')
print("Model and artifacts saved successfully.")

# Section 7: Function for Loading and Predicting
def load_and_predict(new_data):
    model = joblib.load('gdm_model.pkl')
    label_encoder = joblib.load('label_encoder_gdm.pkl')
    scaler = joblib.load('scaler_gdm.pkl')
    saved_columns = joblib.load('columns_gdm.pkl')

    # Feature engineering for new data
    new_data['BP_ratio'] = new_data['Sys_BP'] / new_data['Dia_BP'].replace(0, 1)
    new_data['Age_category'] = pd.cut(new_data['Age'], 
                                     bins=[0, 25, 35, 100], 
                                     labels=[0, 1, 2], 
                                     include_lowest=True)
    new_data['Age_category'] = new_data['Age_category'].cat.codes
    new_data['Age_category'] = new_data['Age_category'].replace(-1, 1)
    new_data['High_OGTT'] = (new_data['OGTT'] > 140).astype(int)
    new_data['BMI_category'] = pd.cut(new_data['BMI'], 
                                     bins=[0, 25, 30, 100], 
                                     labels=[0, 1, 2], 
                                     include_lowest=True)
    new_data['BMI_category'] = new_data['BMI_category'].cat.codes
    new_data['BMI_category'] = new_data['BMI_category'].replace(-1, 1)
    new_data['OGTT_BMI_interaction'] = new_data['OGTT'] * new_data['BMI']
    new_data['Age_OGTT_interaction'] = new_data['Age'] * new_data['OGTT']

    # Ensuring same columns
    new_data = new_data.reindex(columns=saved_columns, fill_value=0)

    # Scaling numeric features
    numeric_cols = ['Age', 'No of Pregnancy', 'BMI', 'HDL', 'Sys_BP', 'Dia_BP', 'OGTT', 
                    'BP_ratio', 'OGTT_BMI_interaction', 'Age_OGTT_interaction']
    new_data[numeric_cols] = scaler.transform(new_data[numeric_cols])

    # Making predictions
    predictions = model.predict(new_data)
    predicted_labels = label_encoder.inverse_transform(predictions)

    return predicted_labels

# Section 8: Example Prediction
new_data = pd.DataFrame({
    'Age': [30], 'No of Pregnancy': [2], 'BMI': [28.0], 'HDL': [50.0], 
    'Family History': [0], 'Large Child or Birth Default': [0], 'PCOS': [0], 
    'Sys_BP': [120], 'Dia_BP': [80], 'OGTT': [130], 'Sedentary Lifestyle': [0], 
    'Prediabetes': [0]
})

predictions = load_and_predict(new_data)
print("\nPrediction for new data:", predictions)