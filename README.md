# 🌙 Lunara: Predictive Diagnostics for Maternal Health

**Lunara** is a machine learning-driven diagnostic platform that provides early risk predictions for maternal health conditions such as **Gestational Diabetes Mellitus (GDM)**, **Preeclampsia**, and general **Maternal Health Risks**. Designed with a cloud-ready architecture and powered by high-performing XGBoost models, Lunara supports timely medical interventions through data-driven insights, targeting global maternal health challenges recognized by the **World Health Organization (WHO)**.

## 🩺 Disease Prediction Summary

Lunara focuses on three core maternal health conditions:

- **Gestational Diabetes Mellitus (GDM)**: Detects elevated blood glucose levels during pregnancy.
- **Preeclampsia**: Predicts high blood pressure complications after 20 weeks of pregnancy.
- **General Maternal Health Risks**: Classifies into low, mid, and high-risk categories based on clinical markers.

> **F1-Scores**:  
> 🔹 GDM — `0.87`  
> 🔹 Preeclampsia — `0.86`  
> 🔹 Maternal Health — `0.85`

## 🚀 Project Objectives

- **🎯 High-Accuracy Predictions**: Deliver reliable model outputs with strong F1-scores via XGBoost and cross-validation.  
- **🔍 Data-Driven Insights**: Utilize feature engineering and preprocessing to maximize signal-to-noise ratio.  
- **☁️ Cloud Scalability**: Architected for Microsoft Azure, ready for real-time deployment.  
- **🔐 Security**: Includes HTTPS, password hashing (Flask-Bcrypt), CSRF protection, and network monitoring.

## 🧠 Model and Tooling Rationale

- **XGBoost Classifiers**: Chosen for imbalanced datasets, fast training, and clinical interpretability.
- **SMOTE & Borderline-SMOTE**: Address underrepresented conditions like preeclampsia effectively.
- **Scikit-learn & Imbalanced-learn**: Robust preprocessing with IQR-based outlier capping and median imputation.
- **Flask + SQLAlchemy**: Lightweight backend with REST endpoints and flexible ORM.
- **PostgreSQL**: Scalable and production-grade RDBMS with foreign key support.
- **Microsoft Azure**: Chosen for deployment due to familiarity and security features.
- **Joblib**: Efficient model serialization for fast inference.

## 🛠️ Tech Stack

| Domain            | Technologies Used                                 |
|-------------------|----------------------------------------------------|
| **ML & Data**     | Python, Pandas, NumPy, XGBoost, Scikit-learn, SMOTE |
| **Backend**       | Flask, SQLAlchemy, Flask-Migrate                   |
| **Frontend**      | HTML, CSS, Jinja2                                  |
| **Database**      | SQLite (dev), PostgreSQL (prod)                    |
| **Security**      | Flask-Bcrypt, HTTPS, CSRF Protection, Wireshark    |
| **Deployment**    | Microsoft Azure (planned)                          |
| **Tools**         | Joblib, Pylint, Flake8                             |

## 🔍 System Architecture


- **ML Layer**: Preprocessing (IQR, SimpleImputer, feature engineering) + XGBoost models.
- **Application Layer**: Flask endpoints (e.g., `/predict_gdm`) integrate pre-trained models.
- **Data Layer**: SQLAlchemy ORM links users to predictions for traceability.
- **Presentation Layer**: Minimal Flask UI with tooltips for medical clarity.

## 🌟 Key Features

- ✔️ Three ML Models for GDM, preeclampsia, and general risk
- ✔️ F1-Scores Above 0.85 through stratified k-fold CV (k=5)
- ✔️ Real-Time Inference under 1.4s
- ✔️ Compact Interface with domain-informed tooltips
- ✔️ CSRF, HTTPS, Bcrypt for secure handling
- ✔️ Caching & Optimizations (e.g., BMI caching for 15% speed-up)

## 🧪 Testing & Validation

- ✅ **Model Validation**: k-fold cross-validation with strong generalization
- ✅ **Performance Testing**:  
  - GDM → 1.2s  
  - Maternal Health → 1.3s  
  - Preeclampsia → 1.4s  
- ✅ **Static Code Analysis**: Pylint score `8/10`, Flake8 compliance
- ✅ **User Acceptance Testing**: Tooltip improvements and mobile usability

## 🚧 Future Enhancements

- 🧬 Ensemble models (e.g., stacking) for higher accuracy
- ⚖️ Load balancing for 100+ concurrent users
- 🗣️ NLP for user feedback analysis
- 📚 Maternal health education modules
- ☁️ Azure Deployment

## 🎓 Professional Relevance

Lunara demonstrates technical and domain expertise across:

- **Machine Learning**: Real-world imbalanced healthcare classification  
- **Data Engineering**: Robust pipelines with IQR outlier handling, median imputation  
- **System Design**: Scalable architecture with cloud readiness and secure backend  
- **Healthcare Application**: Domain knowledge in maternal diagnostics  

**Ideal for roles such as:**

- Data Scientist (Healthcare)  
- Machine Learning Engineer  
- AI Specialist (Medical AI)  

## 📚 References

- [World Health Organization — Maternal Health](https://www.who.int/health-topics/maternal-health)  
- [XGBoost Documentation](https://xgboost.readthedocs.io/en/stable/)  
- [Scikit-learn Model Evaluation](https://scikit-learn.org/stable/modules/model_selection.html)  
- [Imbalanced-learn: SMOTE](https://imbalanced-learn.org/stable/over_sampling.html#smote-adasyn)

## 📝 License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

## 🤝 Contributing

While Lunara is not yet deployed, contributions for cloud deployment, frontend UI improvements, and expanded datasets are welcome. Please open an issue or pull request if you're interested.
