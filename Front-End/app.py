from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file, make_response
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime
from fpdf import FPDF
import io
import ast

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-1234567890')

# Database setup
def init_db():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        full_name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        is_admin INTEGER DEFAULT 0
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        model_type TEXT,
        inputs TEXT,
        prediction TEXT,
        timestamp TEXT
    )''')
    c.execute("SELECT * FROM users WHERE username = 'admin'")
    if not c.fetchone():
        c.execute("INSERT INTO users (username, full_name, email, password, is_admin) VALUES (?, ?, ?, ?, ?)",
                  ('admin', 'Admin User', 'admin@lunara.com', generate_password_hash('admin123'), 1))
    conn.commit()
    conn.close()

init_db()

# Add cache-control headers to prevent caching of sensitive pages
@app.after_request
def add_no_cache_headers(response):
    if not request.path.startswith('/static'):
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
    return response

# Load and Predict functions from model.py files
def load_and_predict_gdm(new_data):
    model = joblib.load('models/Gestational_Diabetic/gdm_model.pkl')
    label_encoder = joblib.load('models/Gestational_Diabetic/label_encoder_gdm.pkl')
    scaler = joblib.load('models/Gestational_Diabetic/scaler_gdm.pkl')
    saved_columns = joblib.load('models/Gestational_Diabetic/columns_gdm.pkl')

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

    new_data = new_data.reindex(columns=saved_columns, fill_value=0)

    numeric_cols = ['Age', 'No of Pregnancy', 'BMI', 'HDL', 'Sys_BP', 'Dia_BP', 'OGTT', 
                    'BP_ratio', 'OGTT_BMI_interaction', 'Age_OGTT_interaction']
    new_data[numeric_cols] = scaler.transform(new_data[numeric_cols])

    predictions = model.predict(new_data)
    predicted_labels = label_encoder.inverse_transform(predictions)

    return predicted_labels[0]

def load_and_predict_maternal(new_data):
    model = joblib.load('models/Maternal_Health/maternal_risk_model.pkl')
    label_encoder = joblib.load('models/Maternal_Health/label_encoder_maternal.pkl')
    scaler = joblib.load('models/Maternal_Health/scaler_maternal.pkl')
    saved_columns = joblib.load('models/Maternal_Health/columns_maternal.pkl')

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

    new_data = new_data.reindex(columns=saved_columns, fill_value=0)

    new_data = scaler.transform(new_data)

    predictions = model.predict(new_data)
    predicted_labels = label_encoder.inverse_transform(predictions)

    return predicted_labels[0]

def load_and_predict_preeclampsia(new_data):
    model = joblib.load('models/Preeclampsia/models/preeclampsia_model_improved.pkl')
    label_encoder = joblib.load('models/Preeclampsia/models/label_encoder_improved.pkl')
    scaler = joblib.load('models/Preeclampsia/models/scaler.pkl')
    saved_columns = joblib.load('models/Preeclampsia/models/columns_improved.pkl')

    new_data['BP_ratio'] = new_data['Systolic BP'] / new_data['Diastolic BP'].replace(0, 1)
    new_data['BP_ratio'] = new_data['BP_ratio'].replace([np.inf, -np.inf], new_data['BP_ratio'].median())

    new_data['BMI_category'] = pd.cut(new_data['BMI'], 
                                     bins=[0, 18.5, 25, 30, 100], 
                                     labels=[0, 1, 2, 3], 
                                     include_lowest=True)
    new_data['BMI_category'] = new_data['BMI_category'].cat.codes
    new_data['BMI_category'] = new_data['BMI_category'].replace(-1, 1)

    new_data = new_data.reindex(columns=saved_columns, fill_value=0)
    new_data = scaler.transform(new_data)

    predictions = model.predict(new_data)
    predicted_labels = label_encoder.inverse_transform(predictions)

    return predicted_labels[0]

# Define custom PDF class for history report
class HistoryPDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Lunara Prediction History Report', 0, 1, 'C')
        self.ln(5)

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(5)

    def chapter_body(self, body):
        self.set_font('Arial', '', 12)
        self.multi_cell(0, 10, body)
        self.ln()

# Generate PDF for prediction history
def generate_history_pdf(predictions):
    pdf = HistoryPDF()
    pdf.add_page()

    pdf.chapter_title('Prediction History')
    if predictions:
        for idx, pred in enumerate(predictions, 1):
            pdf.chapter_title(f'Prediction {idx}')
            pdf.chapter_body(f"Model Type: {pred['model_type']}")
            pdf.chapter_body(f"Prediction: {pred['prediction']}")
            pdf.chapter_body(f"Timestamp: {pred['timestamp']}")
            pdf.chapter_body("Inputs:")
            for key, value in pred['inputs'].items():
                pdf.chapter_body(f"  {key.replace('_', ' ')}: {value}")
            pdf.chapter_title('Clinical Interpretation')
            if pred['model_type'] == "Gestational Diabetes":
                interpretation = (
                    "This indicates a risk for gestational diabetes. Immediate consultation with an obstetrician is recommended."
                    if pred['prediction'] == 'GDM'
                    else "This indicates no immediate concern for gestational diabetes. Continue regular monitoring."
                )
            elif pred['model_type'] == "Maternal Health":
                interpretation = (
                    "This indicates a high-risk pregnancy requiring immediate medical attention."
                    if pred['prediction'] == 'high risk'
                    else "This indicates a moderate risk level; consult a healthcare provider."
                    if pred['prediction'] == 'mid risk'
                    else "This indicates a low risk level, but regular check-ups are recommended."
                )
            else:  # Preeclampsia
                interpretation = (
                    "This indicates a high risk of preeclampsia. Immediate medical consultation is advised."
                    if pred['prediction'] == 'high'
                    else "This indicates a moderate risk level; consult a healthcare provider."
                    if pred['prediction'] == 'mid'
                    else "This indicates a low risk level, but regular monitoring is recommended."
                )
            pdf.chapter_body(interpretation)
            pdf.ln(5)
    else:
        pdf.chapter_body("No prediction history available.")
        pdf.ln(5)

    pdf.chapter_title('Note')
    pdf.chapter_body("This report is generated by Lunara's AI-powered system. Consult a healthcare professional for personalized advice.")

    pdf_buffer = io.BytesIO()
    pdf_bytes = pdf.output(dest='S').encode('latin1')
    pdf_buffer.write(pdf_bytes)
    pdf_buffer.seek(0)
    return pdf_buffer

# Routes
@app.route('/')
def index():
    return redirect(url_for('home'))

@app.route('/home')
def home():
    is_authenticated = 'user_id' in session
    username = session.get('username') if is_authenticated else None
    return render_template('home.html', username=username, is_authenticated=is_authenticated)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        identifier = request.form['identifier'].strip()
        password = request.form['password']
        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username = ? OR email = ?", (identifier, identifier))
        user = c.fetchone()
        conn.close()
        if user and check_password_hash(user[4], password):
            session['user_id'] = user[0]
            session['username'] = user[1]
            session['is_admin'] = user[5]
            flash('Welcome to Lunara!', 'success')
            return redirect(url_for('home'))
        flash('Invalid credentials.', 'error')
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username'].strip()
        full_name = request.form['full_name'].strip()
        email = request.form['email'].strip()
        password = request.form['password']
        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        try:
            c.execute("INSERT INTO users (username, full_name, email, password) VALUES (?, ?, ?, ?)",
                      (username, full_name, email, generate_password_hash(password)))
            conn.commit()
            flash('Account created! Please log in.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username or email already exists.', 'error')
        conn.close()
    return render_template('signup.html')

@app.route('/history')
def history():
    if 'user_id' not in session:
        flash('Please log in to view your prediction history.', 'error')
        return redirect(url_for('login'))
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute("SELECT * FROM predictions WHERE user_id = ?", (session['user_id'],))
    raw_predictions = c.fetchall()
    conn.close()
    predictions = []
    for pred in raw_predictions:
        prediction_dict = {
            'id': pred[0],
            'user_id': pred[1],
            'model_type': pred[2],
            'inputs': ast.literal_eval(pred[3]),
            'prediction': pred[4],
            'timestamp': pred[5]
        }
        predictions.append(prediction_dict)
    return render_template('history.html', predictions=predictions)

@app.route('/download_history_pdf', methods=['POST'])
def download_history_pdf():
    if 'user_id' not in session:
        flash('Please log in to download the history report.', 'error')
        return redirect(url_for('login'))
    try:
        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        c.execute("SELECT * FROM predictions WHERE user_id = ?", (session['user_id'],))
        raw_predictions = c.fetchall()
        conn.close()
        predictions = []
        for pred in raw_predictions:
            prediction_dict = {
                'id': pred[0],
                'user_id': pred[1],
                'model_type': pred[2],
                'inputs': ast.literal_eval(pred[3]),
                'prediction': pred[4],
                'timestamp': pred[5]
            }
            predictions.append(prediction_dict)
        pdf_buffer = generate_history_pdf(predictions)
        response = send_file(
            pdf_buffer,
            as_attachment=True,
            download_name='Lunara_History_Report.pdf',
            mimetype='application/pdf'
        )
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        return response
    except Exception as e:
        flash(f'Failed to generate PDF: {str(e)}', 'error')
        return redirect(url_for('history'))

@app.route('/clear_history', methods=['POST'])
def clear_history():
    if 'user_id' not in session:
        flash('Please log in to clear your history.', 'error')
        return redirect(url_for('login'))
    try:
        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        c.execute("DELETE FROM predictions WHERE user_id = ?", (session['user_id'],))
        conn.commit()
        conn.close()
        flash('Prediction history cleared successfully.', 'success')
        return redirect(url_for('history') + '?cleared=true')
    except Exception as e:
        flash(f'Failed to clear history: {str(e)}', 'error')
        return redirect(url_for('history'))

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        flash('Message sent successfully!', 'success')
    return render_template('contact.html')

@app.route('/predict/gdm', methods=['GET', 'POST'])
def predict_gdm():
    if 'user_id' not in session:
        flash('Please log in to access predictions.', 'error')
        return redirect(url_for('login'))
    
    result = None
    inputs = None
    if request.method == 'POST':
        try:
            age = float(request.form['age'])
            if not (18 <= age <= 50):
                flash('Age must be between 18 and 50.', 'error')
                return render_template('predict_gdm.html', result=None, inputs=None)

            no_pregnancy = int(request.form['no_pregnancy'])
            if not (0 <= no_pregnancy <= 10):
                flash('Number of Pregnancies must be between 0 and 10.', 'error')
                return render_template('predict_gdm.html', result=None, inputs=None)

            bmi = float(request.form['bmi'])
            if not (15.0 <= bmi <= 50.0):
                flash('BMI must be between 15.0 and 50.0.', 'error')
                return render_template('predict_gdm.html', result=None, inputs=None)

            hdl = float(request.form['hdl'])
            if not (20.0 <= hdl <= 100.0):
                flash('HDL must be between 20.0 and 100.0 mg/dL.', 'error')
                return render_template('predict_gdm.html', result=None, inputs=None)

            family_history = int(request.form['family_history'])
            if family_history not in [0, 1]:
                flash('Family History must be 0 (No) or 1 (Yes).', 'error')
                return render_template('predict_gdm.html', result=None, inputs=None)

            large_child = int(request.form['large_child'])
            if large_child not in [0, 1]:
                flash('Large Child or Birth Defect must be 0 (No) or 1 (Yes).', 'error')
                return render_template('predict_gdm.html', result=None, inputs=None)

            pcos = int(request.form['pcos'])
            if pcos not in [0, 1]:
                flash('PCOS must be 0 (No) or 1 (Yes).', 'error')
                return render_template('predict_gdm.html', result=None, inputs=None)

            sys_bp = float(request.form['sys_bp'])
            if not (80.0 <= sys_bp <= 200.0):
                flash('Systolic BP must be between 80.0 and 200.0 mmHg.', 'error')
                return render_template('predict_gdm.html', result=None, inputs=None)

            dia_bp = float(request.form['dia_bp'])
            if not (40.0 <= dia_bp <= 120.0):
                flash('Diastolic BP must be between 40.0 and 120.0 mmHg.', 'error')
                return render_template('predict_gdm.html', result=None, inputs=None)

            ogtt = float(request.form['ogtt'])
            if not (50.0 <= ogtt <= 300.0):
                flash('OGTT must be between 50.0 and 300.0 mg/dL.', 'error')
                return render_template('predict_gdm.html', result=None, inputs=None)

            sedentary = int(request.form['sedentary'])
            if sedentary not in [0, 1]:
                flash('Sedentary Lifestyle must be 0 (No) or 1 (Yes).', 'error')
                return render_template('predict_gdm.html', result=None, inputs=None)

            prediabetes = int(request.form['prediabetes'])
            if prediabetes not in [0, 1]:
                flash('Prediabetes must be 0 (No) or 1 (Yes).', 'error')
                return render_template('predict_gdm.html', result=None, inputs=None)

            inputs = {
                'Age': age,
                'No of Pregnancy': no_pregnancy,
                'BMI': bmi,
                'HDL': hdl,
                'Family History': family_history,
                'Large Child or Birth Default': large_child,
                'PCOS': pcos,
                'Sys_BP': sys_bp,
                'Dia_BP': dia_bp,
                'OGTT': ogtt,
                'Sedentary Lifestyle': sedentary,
                'Prediabetes': prediabetes
            }
            input_df = pd.DataFrame([inputs])
            result = load_and_predict_gdm(input_df)
            conn = sqlite3.connect('database.db')
            c = conn.cursor()
            c.execute("INSERT INTO predictions (user_id, model_type, inputs, prediction, timestamp) VALUES (?, ?, ?, ?, ?)",
                      (session['user_id'], "Gestational Diabetes", str(inputs), result, datetime.now().isoformat()))
            conn.commit()
            conn.close()
        except ValueError:
            flash('Invalid input. Please ensure all fields are numeric where required.', 'error')
            return render_template('predict_gdm.html', result=None, inputs=None)
        except FileNotFoundError:
            flash('Gestational Diabetes model artifacts not found. Please contact support.', 'error')
            return render_template('predict_gdm.html', result=None, inputs=None)

    return render_template('predict_gdm.html', result=result, inputs=inputs)

@app.route('/predict/maternal', methods=['GET', 'POST'])
def predict_maternal():
    if 'user_id' not in session:
        flash('Please log in to access predictions.', 'error')
        return redirect(url_for('login'))
    
    result = None
    inputs = None
    if request.method == 'POST':
        try:
            age = float(request.form['age'])
            if not (10 <= age <= 70):
                flash('Age must be between 10 and 70.', 'error')
                return render_template('predict_maternal.html', result=None, inputs=None)

            systolic_bp = float(request.form['systolic_bp'])
            if not (70.0 <= systolic_bp <= 200.0):
                flash('Systolic BP must be between 70.0 and 200.0 mmHg.', 'error')
                return render_template('predict_maternal.html', result=None, inputs=None)

            diastolic_bp = float(request.form['diastolic_bp'])
            if not (40.0 <= diastolic_bp <= 120.0):
                flash('Diastolic BP must be between 40.0 and 120.0 mmHg.', 'error')
                return render_template('predict_maternal.html', result=None, inputs=None)

            bs = float(request.form['bs'])
            if not (4.0 <= bs <= 20.0):
                flash('Blood Sugar must be between 4.0 and 20.0 mmol/L.', 'error')
                return render_template('predict_maternal.html', result=None, inputs=None)

            heart_rate = int(request.form['heart_rate'])
            if not (40 <= heart_rate <= 120):
                flash('Heart Rate must be between 40 and 120 bpm.', 'error')
                return render_template('predict_maternal.html', result=None, inputs=None)

            inputs = {
                'Age': age,
                'SystolicBP': systolic_bp,
                'DiastolicBP': diastolic_bp,
                'BS': bs,
                'HeartRate': heart_rate
            }
            input_df = pd.DataFrame([inputs])
            result = load_and_predict_maternal(input_df)
            conn = sqlite3.connect('database.db')
            c = conn.cursor()
            c.execute("INSERT INTO predictions (user_id, model_type, inputs, prediction, timestamp) VALUES (?, ?, ?, ?, ?)",
                      (session['user_id'], "Maternal Health", str(inputs), result, datetime.now().isoformat()))
            conn.commit()
            conn.close()
        except ValueError:
            flash('Invalid input. Please ensure all fields are numeric where required.', 'error')
            return render_template('predict_maternal.html', result=None, inputs=None)
        except FileNotFoundError:
            flash('Maternal Health model artifacts not found. Please contact support.', 'error')
            return render_template('predict_maternal.html', result=None, inputs=None)

    return render_template('predict_maternal.html', result=result, inputs=inputs)

@app.route('/predict/preeclampsia', methods=['GET', 'POST'])
def predict_preeclampsia():
    if 'user_id' not in session:
        flash('Please log in to access predictions.', 'error')
        return redirect(url_for('login'))
    
    result = None
    inputs = None
    if request.method == 'POST':
        try:
            gravida = int(request.form['gravida'])
            if not (0 <= gravida <= 10):
                flash('Gravida must be between 0 and 10.', 'error')
                return render_template('predict_preeclampsia.html', result=None, inputs=None)

            parity = int(request.form['parity'])
            if not (0 <= parity <= 10):
                flash('Parity must be between 0 and 10.', 'error')
                return render_template('predict_preeclampsia.html', result=None, inputs=None)

            gestational_age = float(request.form['gestational_age'])
            if not (10.0 <= gestational_age <= 42.0):
                flash('Gestational Age must be between 10.0 and 42.0 weeks.', 'error')
                return render_template('predict_preeclampsia.html', result=None, inputs=None)

            age = float(request.form['age'])
            if not (18 <= age <= 50):
                flash('Age must be between 18 and 50.', 'error')
                return render_template('predict_preeclampsia.html', result=None, inputs=None)

            bmi = float(request.form['bmi'])
            if not (15.0 <= bmi <= 50.0):
                flash('BMI must be between 15.0 and 50.0.', 'error')
                return render_template('predict_preeclampsia.html', result=None, inputs=None)

            diabetes = int(request.form['diabetes'])
            if diabetes not in [0, 1]:
                flash('Diabetes must be 0 (No) or 1 (Yes).', 'error')
                return render_template('predict_preeclampsia.html', result=None, inputs=None)

            hypertension = int(request.form['hypertension'])
            if hypertension not in [0, 1]:
                flash('Hypertension History must be 0 (No) or 1 (Yes).', 'error')
                return render_template('predict_preeclampsia.html', result=None, inputs=None)

            systolic_bp = float(request.form['systolic_bp'])
            if not (70.0 <= systolic_bp <= 200.0):
                flash('Systolic BP must be between 70.0 and 200.0 mmHg.', 'error')
                return render_template('predict_preeclampsia.html', result=None, inputs=None)

            diastolic_bp = float(request.form['diastolic_bp'])
            if not (40.0 <= diastolic_bp <= 120.0):
                flash('Diastolic BP must be between 40.0 and 120.0 mmHg.', 'error')
                return render_template('predict_preeclampsia.html', result=None, inputs=None)

            hb = float(request.form['hb'])
            if not (5.0 <= hb <= 18.0):
                flash('Hemoglobin must be between 5.0 and 18.0 g/dL.', 'error')
                return render_template('predict_preeclampsia.html', result=None, inputs=None)

            fetal_weight = float(request.form['fetal_weight'])
            if not (0.1 <= fetal_weight <= 5.0):
                flash('Fetal Weight must be between 0.1 and 5.0 kg.', 'error')
                return render_template('predict_preeclampsia.html', result=None, inputs=None)

            protein_uria = int(request.form['protein_uria'])
            if protein_uria not in [0, 1]:
                flash('Protein Uria must be 0 (No) or 1 (Yes).', 'error')
                return render_template('predict_preeclampsia.html', result=None, inputs=None)

            amniotic_fluid = float(request.form['amniotic_fluid'])
            if not (5.0 <= amniotic_fluid <= 25.0):
                flash('Amniotic Fluid Levels must be between 5.0 and 25.0 cm.', 'error')
                return render_template('predict_preeclampsia.html', result=None, inputs=None)

            inputs = {
                'gravida': gravida,
                'parity': parity,
                'Gestational_Age_Weeks': gestational_age,
                'Age': age,
                'BMI': bmi,
                'diabetes': diabetes,
                'Hypertension_History': hypertension,
                'Systolic BP': systolic_bp,
                'Diastolic BP': diastolic_bp,
                'HB': hb,
                'Fetal_Weight': fetal_weight,
                'Protein_Uria': protein_uria,
                'Amniotic_Fluid_Levels': amniotic_fluid
            }
            input_df = pd.DataFrame([inputs])
            result = load_and_predict_preeclampsia(input_df)
            conn = sqlite3.connect('database.db')
            c = conn.cursor()
            c.execute("INSERT INTO predictions (user_id, model_type, inputs, prediction, timestamp) VALUES (?, ?, ?, ?, ?)",
                      (session['user_id'], "Preeclampsia", str(inputs), result, datetime.now().isoformat()))
            conn.commit()
            conn.close()
        except ValueError:
            flash('Invalid input. Please ensure all fields are numeric where required.', 'error')
            return render_template('predict_preeclampsia.html', result=None, inputs=None)
        except FileNotFoundError:
            flash('Preeclampsia model artifacts not found. Please contact support.', 'error')
            return render_template('predict_preeclampsia.html', result=None, inputs=None)

    return render_template('predict_preeclampsia.html', result=result, inputs=inputs)

@app.route('/admin')
def admin():
    if 'user_id' not in session or not session.get('is_admin'):
        flash('Access denied. Admin privileges required.', 'error')
        return redirect(url_for('login'))
    try:
        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        c.execute("SELECT * FROM users")
        users = c.fetchall()
        c.execute("SELECT predictions.*, users.username FROM predictions LEFT JOIN users ON predictions.user_id = users.id")
        predictions = c.fetchall()
        c.execute("SELECT model_type, COUNT(*) as count FROM predictions GROUP BY model_type")
        model_counts = c.fetchall()
        conn.close()
        return render_template('admin.html', users=users, predictions=predictions, model_counts=model_counts)
    except Exception as e:
        flash(f'Error loading admin dashboard: {str(e)}', 'error')
        return redirect(url_for('home'))

@app.route('/admin/delete_user/<int:user_id>', methods=['POST'])
def delete_user(user_id):
    if 'user_id' not in session or not session.get('is_admin'):
        flash('Access denied. Admin privileges required.', 'error')
        return redirect(url_for('login'))

    # Prevent admin from deleting their own account
    if user_id == session['user_id']:
        flash('You cannot delete your own account.', 'error')
        return redirect(url_for('admin'))

    try:
        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        # Check if the user exists
        c.execute("SELECT username FROM users WHERE id = ?", (user_id,))
        user = c.fetchone()
        if not user:
            flash('User not found.', 'error')
            conn.close()
            return redirect(url_for('admin'))

        # Delete associated predictions first (to maintain referential integrity)
        c.execute("DELETE FROM predictions WHERE user_id = ?", (user_id,))
        # Delete the user
        c.execute("DELETE FROM users WHERE id = ?", (user_id,))
        conn.commit()
        flash(f'User {user[0]} has been deleted successfully.', 'success')
    except Exception as e:
        conn.rollback()
        flash(f'Error deleting user: {str(e)}', 'error')
    finally:
        conn.close()

    return redirect(url_for('admin'))

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('username', None)
    session.pop('is_admin', None)
    flash('You have been logged out.', 'success')
    response = make_response(redirect(url_for('login')))
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

if __name__ == '__main__':
    app.run(debug=True)