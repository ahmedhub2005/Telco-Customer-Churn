<div dir="ltr">

# 📊 Telco Customer Churn Prediction  

<p align="center">
  <img src="https://img.shields.io/badge/ML-Classification-blue?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Python-3.11-yellow?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Framework-Scikit--learn%20%7C%20XGBoost-green?style=for-the-badge"/>
</p>

---

## 📌 Overview  
Customer churn is a major challenge for telecom companies.  
In this project, the **Telco Customer Churn dataset** was used to build machine learning models that:  
- Predict the likelihood of a customer leaving.  
- Identify **key factors influencing churn**.  
- Provide insights to help businesses improve customer retention.  

---

## 📂 Project Workflow  
🔹 **1. Data Understanding & Cleaning**  
- Handled missing values.  
- Converted categorical variables into numerical features.  
- Engineered new features such as `Tenure Groups`.  

🔹 **2. Exploratory Data Analysis (EDA)**  
- Analyzed customer distribution by contract type, monthly charges, and tenure.  
- Studied the relationship between features and churn behavior.  

🔹 **3. Modeling**  
- Compared multiple models:  
  - Logistic Regression  
  - Random Forest  
  - XGBoost  
- Used **Cross-Validation** to reduce overfitting.  
- Applied **Hyperparameter Tuning** for better performance.  

🔹 **4. Evaluation**  
- Performance metrics: Accuracy, Precision, Recall, F1-score.  
- Visualizations: ROC Curve, Precision-Recall Curve.  
- Feature importance analysis to understand driver factors.  

🔹 **5. Deployment (Future Work)**  
- Prepare the best model for deployment as an API or an interactive app using **Streamlit**.  

---

## ⚙️ Tech Stack  
- **Programming Language:** Python 🐍  
- **Libraries:** pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn, plotly, joblib  
- **Tools:** Jupyter Notebook  

---

## 📈 Key Insights  
- **Best Model:** XGBoost achieved the highest F1-score.  
- **Top Features Influencing Churn:**  
  - Contract Type  
  - Monthly Charges  
  - Tenure  

👉 Businesses should focus on customers with **short-term contracts and high monthly charges** to reduce churn.  

---

## 🔮 Future Improvements  
- Apply advanced balancing techniques such as **SMOTE**.  
- Experiment with other models like **LightGBM** and **CatBoost**.  
- Deploy the model as an interactive app (Streamlit or Flask).  
- Add interactive dashboards using **Tableau** or **Power BI**.  

---

## 👤 Author  
**Ahmed Hamdy**  

</div>

🔗 GitHub
