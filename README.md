Lunara: Predictive Diagnostics for Maternal Health
Overview
Lunara is a machine learning-driven application designed for predictive diagnostics in maternal health, focusing on early detection of gestational diabetes mellitus (GDM), preeclampsia, and general maternal health risks. Leveraging advanced XGBoost classifiers, sophisticated preprocessing pipelines, and a cloud-ready architecture, Lunara delivers high-accuracy risk predictions to support timely medical interventions. Addressing critical global maternal health challenges highlighted by the World Health Organization, the platform emphasizes robust model performance and scalability, though it is currently not deployed. This project showcases expertise in machine learning, data engineering, and system design, making it a compelling portfolio piece for data science and machine learning engineering roles in healthcare.
Disease Prediction Summary
Lunara predicts three maternal health conditions:

Gestational Diabetes Mellitus (GDM): A condition of elevated blood sugar during pregnancy, increasing risks of complications like macrosomia and cesarean delivery. Prevention includes maintaining a balanced diet, regular exercise, and monitoring blood glucose levels. Precautions involve regular prenatal check-ups and early screening for at-risk individuals (e.g., those with family history or high BMI).
Preeclampsia: Characterized by high blood pressure and potential organ damage after 20 weeks of pregnancy, risking maternal and fetal health. Prevention includes low-dose aspirin for high-risk cases, managing blood pressure, and a healthy lifestyle. Precautions involve monitoring blood pressure and protein in urine during prenatal visits.
General Maternal Health Risks: Encompasses low, mid, and high-risk categories based on factors like age, blood pressure, and blood sugar. Prevention focuses on healthy weight management, stress reduction, and regular medical check-ups. Precautions include early risk assessment and adherence to medical advice.

Lunara’s machine learning models analyze clinical features (e.g., BMI, blood pressure, OGTT) to predict risks with F1-scores above 0.85, enabling proactive healthcare interventions.
Project Description
Lunara tackles maternal health issues like GDM and preeclampsia, which contribute significantly to global morbidity and mortality due to delayed diagnosis. Its core strength lies in robust XGBoost models optimized for high accuracy, supported by advanced data preprocessing and feature engineering. Designed for scalability on Microsoft Azure, Lunara ensures secure data handling and is architected for production-grade deployment. The project prioritizes machine learning innovation over aesthetic design, aligning with industry needs for impactful AI solutions in healthcare.
Objectives

High-Accuracy Predictions: Develop XGBoost classifiers to predict GDM, preeclampsia, and maternal health risks with robust performance metrics.
Data-Driven Insights: Implement sophisticated preprocessing and feature engineering to maximize model reliability.
Scalability: Architect for cloud-based deployment to handle large-scale data processing and user concurrency.
Security: Ensure compliance with data protection standards through encryption and secure authentication.

Model and Tool Selection Rationale
Lunara’s design choices prioritize performance, interpretability, and scalability for healthcare applications:

XGBoost Classifiers: Chosen over other models (e.g., Random Forest, SVM, Neural Networks) due to their superior handling of imbalanced datasets, high predictive accuracy, and efficiency with small-to-medium-sized healthcare datasets. XGBoost’s gradient boosting framework excels in capturing complex feature interactions, critical for clinical data, and provides feature importance insights for interpretability, which is essential in healthcare for trust and validation. For instance, XGBoost achieved F1-scores of 0.87 (GDM), 0.85 (maternal health), and 0.86 (preeclampsia), outperforming Random Forest (0.82-0.84) in internal tests.
SMOTE and Borderline-SMOTE: Selected to address class imbalances in maternal health datasets (e.g., fewer GDM or preeclampsia cases). Unlike random oversampling, SMOTE generates synthetic samples to avoid overfitting, while Borderline-SMOTE focuses on minority class boundaries for improved model generalization, particularly effective for preeclampsia’s multiclass problem.
Scikit-learn (StandardScaler, SimpleImputer): Used for preprocessing due to its robust, industry-standard implementation for scaling and imputation. SimpleImputer with median strategy was chosen over mean to handle skewed clinical data (e.g., blood pressure), ensuring robustness against outliers.
Pandas and NumPy: Preferred for data manipulation due to their efficiency, flexibility, and widespread use in data science pipelines, enabling seamless integration with Scikit-learn and XGBoost.
Flask and SQLAlchemy: Selected for the backend over alternatives like Django due to Flask’s lightweight nature, sufficient for a functional interface, and SQLAlchemy’s flexibility for database management. This choice prioritizes rapid development and integration with machine learning models over complex web frameworks.
PostgreSQL (Production): Chosen over MySQL for its robustness in handling complex queries and scalability in production environments, critical for healthcare data management.
Microsoft Azure: Planned for deployment due to its scalability, security features, and support for machine learning workflows, compared to AWS or GCP, which were considered but not selected due to project-specific Azure familiarity.
Joblib: Used for model serialization over Pickle due to its efficiency with large NumPy arrays, ensuring faster model loading for real-time predictions.These choices reflect a balance of performance, interpretability, and deployment readiness, tailored to healthcare’s stringent requirements.

Key Features

Machine Learning Models: Three XGBoost classifiers for GDM, maternal health risks, and preeclampsia, achieving F1-scores of 0.87 (GDM), 0.85 (maternal health), and 0.86 (preeclampsia) via k-fold cross-validation. Models leverage SMOTE and Borderline-SMOTE for class imbalance correction.
Data Preprocessing Pipeline: Includes IQR-based outlier capping, feature engineering (e.g., BP_ratio, High_BS, interaction terms like OGTT_BMI), and SimpleImputer for missing data, ensuring robust input quality.
Model Optimization: Achieves prediction times of 1.2s (GDM), 1.3s (maternal health), and 1.4s (preeclampsia), with optimizations like BMI calculation caching (15% time reduction).
Functional Interface: Flask-based interface with compact forms and tooltips (e.g., "Systolic BP: 70-120 mmHg") for medical input guidance, ensuring usability.
Database Management: Utilizes SQLite (development) and PostgreSQL (production) with SQLAlchemy for storing user data and prediction records.
Security: Implements CSRF protection, Flask-Bcrypt for password hashing, and HTTPS encryption, with network monitoring via Wireshark.
Cloud-Ready Architecture: Designed for Azure deployment to support scalable model inference, though currently not deployed.

Technology Stack

Machine Learning: Python, Pandas, NumPy, Scikit-learn (StandardScaler, SimpleImputer), XGBoost, Imbalanced-learn (SMOTE, Borderline-SMOTE), Joblib
Backend: Flask, SQLAlchemy, Flask-Migrate
Frontend: HTML, CSS, Jinja2 (minimalistic, functional design)
Database: SQLite (development), PostgreSQL (production)
DevOps & Tools: Microsoft Azure (planned), Pylint, Flake8, Wireshark
Security: Flask-Bcrypt, CSRF protection, HTTPS

System Architecture
Lunara’s architecture prioritizes machine learning and data processing:

Machine Learning Layer: Core component with three XGBoost models for GDM, maternal health, and preeclampsia. Preprocessing pipelines handle outlier capping, feature engineering (e.g., BP_ratio, High_BS, interaction terms), and SimpleImputer for missing data.
Application Layer: Flask manages routing (e.g., /predict_gdm) and integrates serialized models (e.g., gdm_model.pkl) for real-time predictions.
Data Layer: SQL database with SQLAlchemy ORM, using foreign keys to link predictions to user IDs for traceability.
Presentation Layer: Minimal Flask-based interface with Jinja2 templates, featuring compact forms and a fixed navbar.

Key Achievements

Model Performance: Achieved F1-scores of 0.87 (GDM), 0.85 (maternal health), and 0.86 (preeclampsia) through k-fold cross-validation (k=5), with robust handling of noisy and missing data.
Optimization: Reduced preprocessing time by 15% through caching and streamlined model inference for sub-2-second predictions.
Code Quality: Improved codebase to achieve a Pylint score of 8/10 by resolving 25 Flake8 issues and 15 Pylint issues.
Testing: Validated model robustness with noisy inputs, achieving 80 concurrent user support (target: 100). User feedback informed improvements in input tooltips and result clarity.

Testing

Model Validation: Conducted k-fold cross-validation (k=5) to achieve F1-scores above 0.85, with stress tests for noisy and incomplete inputs.
Performance Testing: Recorded prediction times of 1.2s (GDM), 1.3s (maternal health), and 1.4s (preeclampsia), with form submission latency under 0.8s.
Static Code Analysis: Used Flake8 and Pylint to optimize preprocessing pipelines, achieving a Pylint score of 8/10.
User Acceptance Testing: Incorporated feedback to enhance medical input tooltips, mobile compatibility, and prediction explanations.

Future Enhancements

Advanced Modeling: Integrate ensemble methods (e.g., stacking) and additional features (e.g., genetic markers) to improve accuracy.
Scalability: Implement load balancing and distributed computing for 100+ concurrent users upon deployment.
NLP Integration: Add natural language processing to analyze user feedback.
User Education: Develop FAQ and educational content for maternal health.
Deployment: Deploy on Azure to enable real-world usage and validate scalability.

Professional Relevance
Lunara demonstrates advanced skills critical for data science and machine learning engineering roles:

Machine Learning Expertise: Proficiency in XGBoost, SMOTE, and preprocessing pipelines, showcasing model development and optimization for healthcare.
Data Engineering: Strong skills in feature engineering, data cleaning, and handling imbalanced datasets, essential for real-world data science.
System Design: Scalable, cloud-ready architecture aligning with industry standards for AI solutions.
Code Quality: Adherence to best practices, validated by Pylint and Flake8, ensuring production-ready code.
Healthcare Impact: Application of machine learning to address maternal health challenges, demonstrating domain knowledge and social impact.

This project highlights transferable skills for roles such as Data Scientist, Machine Learning Engineer, or AI Specialist in healthcare, emphasizing technical expertise and problem-solving.
Why Lunara Stands Out
I am enthusiastic about the Lunara project for the following reasons:

Impactful Application: It addresses critical maternal health issues with the potential to improve outcomes through early detection, aligning with global health priorities.
Technical Rigor: The project demonstrates advanced machine learning techniques, including hyperparameter tuning, SMOTE for class imbalance, and feature engineering, showcasing my ability to tackle complex data challenges.
Model Selection: The use of XGBoost over alternatives like Random Forest or SVM reflects a deliberate choice for high accuracy and interpretability, critical for healthcare applications.
Scalability and Professionalism: Designed with production-grade scalability and security, Lunara reflects my understanding of industry requirements for deployable AI solutions.
End-to-End Development: From data preprocessing to model training and integration, the project highlights my ability to manage a complete machine learning pipeline, making it a compelling addition to my portfolio for data science roles.

References

World Health Organization. (n.d.) Maternal health. https://www.who.int/health-topics/maternal-health
Scikit-learn. (n.d.) Model selection and evaluation. https://scikit-learn.org/stable/modules/model_selection.html
XGBoost. (n.d.) Introduction to XGBoost. https://xgboost.readthedocs.io/en/stable/
Imbalanced-learn. (n.d.) Synthetic Minority Over-sampling Technique (SMOTE). https://imbalanced-learn.org/stable/

License
This project is licensed under the MIT License. See the LICENSE file for details.