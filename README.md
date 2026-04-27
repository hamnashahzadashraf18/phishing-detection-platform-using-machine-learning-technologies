This project is a Phishing URL Detection System developed as a Final Year Project 
for the Bachelor of Software Engineering degree. It uses Machine Learning to 
automatically detect whether a website URL is phishing (dangerous) or legitimate 
(safe) by extracting 22 features directly from the URL string.

Eight machine learning algorithms were trained and compared on a dataset of 11,055 
labelled URLs from Kaggle — Logistic Regression, SVM, Decision Tree, Random Forest, 
KNN, XGBoost, CatBoost, and Naive Bayes. XGBoost achieved the best performance 
with 97.15% accuracy and 97.46% F1-Score and was selected as the final model.

The complete web application is built using Streamlit and includes a user login and 
signup system, a URL checker that gives instant predictions with confidence percentage 
and feature breakdown, a personal dashboard showing search history, and a model 
comparison chart displaying all 8 algorithms. User data is stored in a SQLite database.

The project also includes a special URL shortener detection override for services 
like bit.ly and goo.gl, and a confidence threshold that flags uncertain predictions 
as suspicious rather than giving a false definitive answer.

Tech Stack: Python, XGBoost, Scikit-learn, Streamlit, SQLite, Pandas, NumPy, Joblib
