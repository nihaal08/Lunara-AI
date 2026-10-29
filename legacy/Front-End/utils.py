"""
Utility functions for LUNARA-AI application
Includes database operations, security helpers, and model predictions
"""

import sqlite3
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from functools import wraps
from flask import session, redirect, url_for, render_template

class DatabaseManager:
    """Manages database operations"""
    
    def __init__(self, db_path):
        self.db_path = db_path
    
    def get_connection(self):
        """Get database connection"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def init_db(self):
        """Initialize database with required tables"""
        conn = self.get_connection()
        c = conn.cursor()
        
        # Users table
        c.execute('''CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            full_name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            is_admin INTEGER DEFAULT 0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            last_login TEXT,
            is_active INTEGER DEFAULT 1
        )''')
        
        # Predictions table
        c.execute('''CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            model_type TEXT NOT NULL,
            inputs TEXT NOT NULL,
            prediction TEXT NOT NULL,
            confidence REAL,
            timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )''')
        
        # Audit log table
        c.execute('''CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            action TEXT NOT NULL,
            details TEXT,
            ip_address TEXT,
            timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )''')
        
        conn.commit()
        conn.close()

class ModelPredictor:
    """Handles model predictions for all three health conditions"""
    
    def __init__(self, models_path):
        self.models_path = models_path
    
    def predict_gdm(self, new_data):
        """Predict Gestational Diabetes Mellitus"""
        try:
            model = joblib.load(os.path.join(self.models_path, 'Gestational_Diabetic', 'gdm_model.pkl'))
            label_encoder = joblib.load(os.path.join(self.models_path, 'Gestational_Diabetic', 'label_encoder_gdm.pkl'))
            scaler = joblib.load(os.path.join(self.models_path, 'Gestational_Diabetic', 'scaler_gdm.pkl'))
            saved_columns = joblib.load(os.path.join(self.models_path, 'Gestational_Diabetic', 'columns_gdm.pkl'))
            
            # Feature engineering
            new_data['BP_ratio'] = new_data['Sys_BP'] / new_data['Dia_BP'].replace(0, 1)
            new_data['Age_category'] = pd.cut(new_data['Age'], 
                                             bins=[0, 25, 35, 100], 
                                             labels=[0, 1, 2], 
                                             include_lowest=True).cat.codes
            new_data['Age_category'] = new_data['Age_category'].replace(-1, 1)
            new_data['High_OGTT'] = (new_data['OGTT'] > 140).astype(int)
            new_data['BMI_category'] = pd.cut(new_data['BMI'], 
                                             bins=[0, 25, 30, 100], 
                                             labels=[0, 1, 2], 
                                             include_lowest=True).cat.codes
            new_data['BMI_category'] = new_data['BMI_category'].replace(-1, 1)
            new_data['OGTT_BMI_interaction'] = new_data['OGTT'] * new_data['BMI']
            new_data['Age_OGTT_interaction'] = new_data['Age'] * new_data['OGTT']
            
            new_data = new_data.reindex(columns=saved_columns, fill_value=0)
            
            numeric_cols = ['Age', 'No of Pregnancy', 'BMI', 'HDL', 'Sys_BP', 'Dia_BP', 'OGTT', 
                           'BP_ratio', 'OGTT_BMI_interaction', 'Age_OGTT_interaction']
            new_data[numeric_cols] = scaler.transform(new_data[numeric_cols])
            
            predictions = model.predict(new_data)
            predicted_labels = label_encoder.inverse_transform(predictions)
            
            return predicted_labels[0]
        except Exception as e:
            raise Exception(f"Error in GDM prediction: {str(e)}")
    
    def predict_maternal(self, new_data):
        """Predict Maternal Health Risk"""
        try:
            model = joblib.load(os.path.join(self.models_path, 'Maternal_Health', 'maternal_risk_model.pkl'))
            label_encoder = joblib.load(os.path.join(self.models_path, 'Maternal_Health', 'label_encoder_maternal.pkl'))
            scaler = joblib.load(os.path.join(self.models_path, 'Maternal_Health', 'scaler_maternal.pkl'))
            saved_columns = joblib.load(os.path.join(self.models_path, 'Maternal_Health', 'columns_maternal.pkl'))
            
            # Feature engineering
            new_data['BP_ratio'] = new_data['SystolicBP'] / new_data['DiastolicBP'].replace(0, 1)
            new_data['Age_category'] = pd.cut(new_data['Age'], 
                                             bins=[0, 18, 35, 100], 
                                             labels=[0, 1, 2], 
                                             include_lowest=True).cat.codes
            
            new_data = new_data.reindex(columns=saved_columns, fill_value=0)
            
            numeric_cols = ['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate', 'BP_ratio']
            new_data[numeric_cols] = scaler.transform(new_data[numeric_cols])
            
            predictions = model.predict(new_data)
            predicted_labels = label_encoder.inverse_transform(predictions)
            
            return predicted_labels[0]
        except Exception as e:
            raise Exception(f"Error in Maternal Health prediction: {str(e)}")
    
    def predict_preeclampsia(self, new_data):
        """Predict Preeclampsia Risk"""
        try:
            model = joblib.load(os.path.join(self.models_path, 'Preeclampsia', 'preeclampsia_model.pkl'))
            scaler = joblib.load(os.path.join(self.models_path, 'Preeclampsia', 'scaler_preeclampsia.pkl'))
            saved_columns = joblib.load(os.path.join(self.models_path, 'Preeclampsia', 'columns_preeclampsia.pkl'))
            
            new_data = new_data.reindex(columns=saved_columns, fill_value=0)
            
            numeric_cols = saved_columns
            new_data[numeric_cols] = scaler.transform(new_data[numeric_cols])
            
            predictions = model.predict(new_data)
            
            return 'Yes' if predictions[0] == 1 else 'No'
        except Exception as e:
            raise Exception(f"Error in Preeclampsia prediction: {str(e)}")


def login_required(f):
    """Decorator to require login"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('user_id'):
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


def admin_required(f):
    """Decorator to require admin role"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('user_id') or not session.get('is_admin'):
            return render_template('error.html', message='Unauthorized - Admin access required'), 403
        return f(*args, **kwargs)
    return decorated_function
