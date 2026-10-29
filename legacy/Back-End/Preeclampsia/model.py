# Import necessary libraries
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

# ==========================
# Data Loading and Cleaning with additional steps
# ==========================

# Load dataset
df = pd.read_csv("Preeclampsia.csv")

# 1. Remove irrelevant column
df = df.drop(columns=["Unnamed: 12"], errors='ignore')

# 2. Strip whitespace from column names
df.columns = df.columns.str.strip()

# 3. Rename columns for clarity
df.rename(columns={
    "Age (yrs)": "Age",
    "BMI  [kg/m²]": "BMI",
    "History of hypertension (y/n)": "Hypertension_History",
    "gestational age (weeks)": "Gestational_Age_Weeks",
    "fetal weight(kgs)": "Fetal_Weight",
    "Protien Uria": "Protein_Uria",
    "Uterine Artery Doppler Resistance Index (RI)": "Uterine_RI",
    "Uterine Artery Doppler Pulsatility Index (PI": "Uterine_PI",
    "amniotic fluid levels(cm)": "Amniotic_Fluid_Levels"
}, inplace=True)

# 4. Convert numeric columns to appropriate types
numeric_cols = [
    'gravida', 'parity', 'Gestational_Age_Weeks', 'Age', 'BMI', 
    'Systolic BP', 'Diastolic BP', 'HB', 'Fetal_Weight', 
    'Uterine_RI', 'Uterine_PI', 'Amniotic_Fluid_Levels'
]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 5. Handle missing values
# Drop rows with missing values in critical columns
df.dropna(subset=['Age', 'Systolic BP', 'Diastolic BP', 'Fetal_Weight'], inplace=True)
# Fill remaining NaNs with column mean
df.fillna(df.mean(numeric_only=True), inplace=True)

# 6. Reset index
df.reset_index(drop=True, inplace=True)

# 7. Confirm cleanup
print(df.info())
print(df.head())

# ==========================
# Continue with original pipeline
# ==========================

# Remove duplicate header if exists
df = df[df['gravida'] != 'gravida']

# Drop other irrelevant columns if still needed
columns_to_drop = ['Uterine_RI', 'Uterine_PI']
df = df.drop(columns=columns_to_drop, errors='ignore')

# Strip whitespace from column names again if needed
df.columns = df.columns.str.strip()

# Rename columns for consistency (if not already done)
# (Already done above, but you can ensure again if needed)

# Check data types
print(df.dtypes)

# Check nulls
print(df.isnull().sum())

# Handle 'Uterine_PI' if present
# (skip if not relevant or already handled)

# Section 2: Preprocessing Numeric Columns
numeric_columns = [
    'gravida', 'parity', 'Gestational_Age_Weeks', 'Age',
    'BMI', 'Systolic BP', 'Diastolic BP', 'HB',
    'Fetal_Weight', 'Amniotic_Fluid_Levels'
]
from sklearn.impute import SimpleImputer

num_imputer = SimpleImputer(strategy='median')
for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        # Cap outliers using IQR
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df[col] = df[col].clip(lower=Q1 - 1.5 * IQR, upper=Q3 + 1.5 * IQR)
# Impute missing values
df[numeric_columns] = num_imputer.fit_transform(df[numeric_columns])

# Section 3: Preprocessing Binary Columns
binary_columns = ['diabetes', 'Hypertension_History', 'Protein_Uria']
bin_imputer = SimpleImputer(strategy='most_frequent')
for col in binary_columns:
    if col in df.columns:
        df[col] = df[col].apply(lambda x: 1 if str(x).lower() in ['y', '1', 'yes', 'true']
                              else 0 if str(x).lower() in ['n', '0', 'no', 'false']
                              else np.nan)
df[binary_columns] = bin_imputer.fit_transform(df[binary_columns])

# Section 4: Feature Engineering
# BP ratio
df['BP_ratio'] = df['Systolic BP'] / df['Diastolic BP'].replace(0, 1)
df['BP_ratio'] = df['BP_ratio'].replace([np.inf, -np.inf], df['BP_ratio'].median())

# BMI category
df['BMI_category'] = pd.cut(df['BMI'], 
                             bins=[0, 18.5, 25, 30, 100], 
                             labels=[0, 1, 2, 3], 
                             include_lowest=True)
df['BMI_category'] = df['BMI_category'].cat.codes
df['BMI_category'] = df['BMI_category'].replace(-1, 1)

# Section 5: Final data checks & encoding
if df.isna().any().any():
    print("NaNs found in columns:", df.columns[df.isna().any()].tolist())
    df = df.fillna(df.median(numeric_only=True))
# Encode target variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Risk_level'] = le.fit_transform(df['Risk_level'])

# Section 6: Prepare features and target
X = df.drop('Risk_level', axis=1)
y = df['Risk_level']
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
if X.isna().any().any():
    print("NaNs in X after preprocessing:", X.columns[X.isna().any()].tolist())

# Section 7: Model training
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

xgb_model = XGBClassifier(random_state=42, eval_metric='mlogloss')
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1]
}
grid_search = GridSearchCV(xgb_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)
cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='accuracy')
print("Cross-Validation Accuracy: {:.2f} ± {:.2f}".format(cv_scores.mean(), cv_scores.std()))

# Evaluate on test
y_pred = best_model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Section 8: Save model and artifacts
joblib.dump(best_model, 'preeclampsia_model_improved.pkl')
joblib.dump(le, 'label_encoder_improved.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(X_train.columns, 'columns_improved.pkl')
print("Model and artifacts saved.")

# Section 9: Load and predict function
def load_and_predict(new_data, model_path='preeclampsia_model_improved.pkl', 
                     le_path='label_encoder_improved.pkl', 
                     scaler_path='scaler.pkl', 
                     columns_path='columns_improved.pkl'):
    model = joblib.load(model_path)
    label_encoder = joblib.load(le_path)
    scaler = joblib.load(scaler_path)
    saved_columns = joblib.load(columns_path)
    
    # Feature engineering
    new_data['BP_ratio'] = new_data['Systolic BP'] / new_data['Diastolic BP'].replace(0, 1)
    new_data['BP_ratio'] = new_data['BP_ratio'].replace([np.inf, -np.inf], new_data['BP_ratio'].median())

    new_data['BMI_category'] = pd.cut(new_data['BMI'], 
                                     bins=[0, 18.5, 25, 30, 100], 
                                     labels=[0, 1, 2, 3], 
                                     include_lowest=True)
    new_data['BMI_category'] = new_data['BMI_category'].cat.codes
    new_data['BMI_category'] = new_data['BMI_category'].replace(-1, 1)

    # Reindex columns
    new_data = new_data.reindex(columns=saved_columns, fill_value=0)
    # Scale
    new_data = scaler.transform(new_data)
    # Predict
    predictions = model.predict(new_data)
    predicted_labels = label_encoder.inverse_transform(predictions)
    return predicted_labels

# Section 10: Example prediction
new_data = pd.DataFrame({
    'gravida': [3], 'parity': [1], 'Gestational_Age_Weeks': [22.2], 'Age': [25],
    'BMI': [21.0], 'diabetes': [0], 'Hypertension_History': [0],
    'Systolic BP': [110], 'Diastolic BP': [70], 'HB': [9.3], 'Fetal_Weight': [0.501],
    'Protein_Uria': [0], 'Amniotic_Fluid_Levels': [10.0]
})

predictions = load_and_predict(new_data)
print("\nPrediction for new data:", predictions)