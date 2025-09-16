# 📊 Telco Customer Churn Prediction  

<p align="center">
  <img src="https://img.shields.io/badge/ML-Classification-blue?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Python-3.11-yellow?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Framework-Scikit--learn%20%7C%20XGBoost-green?style=for-the-badge"/>
</p>

---

## 📌 Overview  
Customer churn (ترك العملاء للشركة) يمثل تحديًا كبيرًا لشركات الاتصالات.  
في هذا المشروع تم استخدام **بيانات Telco Customer Churn** لبناء نماذج تعلم آلي تساعد على:  
- التنبؤ بمدى احتمالية مغادرة العميل.  
- فهم **العوامل الأكثر تأثيرًا** على churn.  
- تقديم رؤى تساعد الشركات على تحسين خدمة العملاء وتقليل نسبة المغادرين.  

---

## 📂 Project Workflow  
🔹 **1. Data Understanding & Cleaning**  
- التعامل مع القيم المفقودة.  
- تحويل المتغيرات الفئوية (Categorical) إلى متغيرات رقمية.  
- استخراج ميزات جديدة مثل `Tenure Groups`.  

🔹 **2. Exploratory Data Analysis (EDA)**  
- تحليل توزيعات العملاء حسب العقود، الفواتير الشهرية، مدة الخدمة.  
- دراسة ارتباط السمات بسلوك churn.  

🔹 **3. Modeling**  
- مقارنة عدة نماذج:  
  - Logistic Regression  
  - Random Forest  
  - XGBoost  
- استخدام **Cross-Validation** للحد من الـ Overfitting.  
- ضبط المعاملات (Hyperparameter Tuning).  

🔹 **4. Evaluation**  
- مقاييس الأداء: Accuracy, Precision, Recall, F1-score.  
- رسم منحنيات ROC و Precision-Recall.  
- تحليل **Feature Importance**.  

🔹 **5. Deployment (Future Work)**  
- تجهيز النموذج كـ API أو واجهة تفاعلية باستخدام **Streamlit**.  

---

## ⚙️ Tech Stack  
- **Programming Language:** Python 🐍  
- **Libraries:** pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn, plotly, joblib  
- **Tools:** Jupyter Notebook  

---


 

أفضل نموذج حقق F1-score أعلى كان XGBoost.

السمات الأكثر تأثيرًا على churn:

نوع العقد (Contract Type)

قيمة الفاتورة الشهرية (MonthlyCharges)

مدة الخدمة (Tenure)

📈 الشركات يمكنها التركيز على العملاء أصحاب العقود القصيرة والفواتير العالية لتقليل churn.

🔮 Future Improvements

استخدام تقنيات موازنة البيانات مثل SMOTE.

تجربة نماذج أخرى مثل LightGBM & CatBoost.

نشر النموذج كـ تطبيق تفاعلي (Streamlit أو Flask).

إضافة تقارير تفاعلية باستخدام Tableau أو PowerBI.

👤 Author

Ahmed Hamdy

🔗 GitHub
