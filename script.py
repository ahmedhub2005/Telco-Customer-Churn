# ========== Standard Libraries ==========
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ========== Scikit-learn ==========
from sklearn.model_selection import train_test_split, learning_curve, validation_curve
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Metrics & Evaluation
from sklearn import metrics
from sklearn.metrics import (
    accuracy_score,
    mean_squared_error,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    ConfusionMatrixDisplay,
    roc_auc_score,
    classification_report

)

# ========== Imbalanced Data ==========
from imblearn.over_sampling import SMOTE

# ========== XGBoost ==========
from xgboost import XGBClassifier

# ========== Save Model==========
import joblib



## Read the Dataset
TRAIN_PATH = os.path.join(os.getcwd(),'telco_churn' , 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
data = pd.read_csv(TRAIN_PATH)
data.head()
data.describe()
data.dtypes
data.info()
sns.heatmap(data.isna())
import plotly.graph_objects as go
from plotly.subplots import make_subplots
g_labels = ['Male', 'Female']
c_labels = ['No', 'Yes']
fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])
fig.add_trace(go.Pie(labels=g_labels, values=data['gender'].value_counts(), name="Gender"),
              1, 1)
fig.add_trace(go.Pie(labels=c_labels, values=data['Churn'].value_counts(), name="Churn"),
              1, 2)


fig.update_traces(hole=.4, hoverinfo="label+percent+name", textfont_size=16)

fig.update_layout(
    title_text="Gender and Churn Distributions",

    annotations=[dict(text='Gender', x=0.19, y=0.5, font_size=20, showarrow=False),
                 dict(text='Churn', x=0.81, y=0.5, font_size=20, showarrow=False)])
fig.show()

import plotly.express as px
fig = px.histogram(data, x="Churn", color="Contract", barmode="group", title="<b>Customer contract distribution<b>")
fig.update_layout(width=700, height=500, bargap=0.1)
fig.show()

fig = px.histogram(data, x="Churn", color="PaymentMethod", title="<b>Customer Payment Method distribution w.r.t. Churn</b>")
fig.update_layout(width=700, height=500, bargap=0.1)
fig.show()
sns.set_context("paper",font_scale=1.1)
ax = sns.kdeplot(data.MonthlyCharges[(data["Churn"] == 'No') ],
                color="Red", shade = True);
ax = sns.kdeplot(data.MonthlyCharges[(data["Churn"] == 'Yes') ],
                ax =ax, color="Blue", shade= True);
ax.legend(["Not Churn","Churn"],loc='upper right');
ax.set_ylabel('Density');
ax.set_xlabel('Monthly Charges');
ax.set_title('Distribution of monthly charges by churn');

z_scores = np.abs(stats.zscore(data['MonthlyCharges']))
outliers = data[z_scores > 3]
print(outliers.shape)
data=data.drop(["customerID"] , axis=1)
data["Churn"] = pd.get_dummies(data["Churn"], drop_first=True)


y=data["Churn"]

x=data.drop(["Churn"] , axis=1)

X_train , X_test , y_train , y_test =train_test_split(x ,y , test_size=0.3,random_state=42 , stratify=y)


import pandas as pd

# نفترض إن عندك DataFrame اسمه data
categorical_columns = data.select_dtypes(include=['object']).columns

print("Categorical columns:")
print(categorical_columns)

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# تأكد إن TotalCharges أرقام
X_train['TotalCharges'] = pd.to_numeric(X_train['TotalCharges'], errors='coerce')
X_test['TotalCharges'] = pd.to_numeric(X_test['TotalCharges'], errors='coerce')

# numeric & categorical
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
categ_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',
              'PhoneService', 'MultipleLines', 'InternetService',
              'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
              'TechSupport', 'StreamingTV', 'StreamingMovies',
              'Contract', 'PaperlessBilling', 'PaymentMethod']
ready_cols = list(set(X_train.columns) - set(num_cols) - set(categ_cols))

# build ColumnTransformer (بدون DataFrameSelector ولا FeatureUnion)
all_pipeline = ColumnTransformer(transformers=[
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), num_cols),

    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(drop='first', sparse_output=False))
    ]), categ_cols),

    ('ready', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent'))
    ]), ready_cols)
])

# apply
X_train_final = all_pipeline.fit_transform(X_train)
X_test_final = all_pipeline.transform(X_test)

print(X_train_final.shape, X_test_final.shape)

# ==============================
# 5. Handle Imbalanced Data (SMOTE)
# ==============================
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X_train_final, y_train)

print("Before SMOTE:", y_train.value_counts().to_dict())
print("After SMOTE:", pd.Series(y_resampled).value_counts().to_dict())



# ==============================
# 6. Train Models
# ==============================
models = {
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"),
    "LogisticRegression": LogisticRegression(max_iter=500, class_weight="balanced"),
    "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42),
}

for name, model in models.items():
    model.fit(X_resampled, y_resampled)
    y_pred = model.predict(X_test_final)
    print(f"\n{name}")
    print(classification_report(y_test, y_pred))
    print("AUC:", roc_auc_score(y_test, model.predict_proba(X_test_final)[:, 1]))

# ==============================
# 10. Feature Importance (XGBoost)
# ==============================
model_xgb = XGBClassifier(eval_metric="logloss", random_state=42)
model_xgb.fit(X_resampled, y_resampled)

# Get feature names
feature_names = (
    num_cols +
    list(all_pipeline.named_transformers_['cat'].named_steps['ohe'].get_feature_names_out(categ_cols)) +
    ready_cols
)

importances = model_xgb.feature_importances_
indices = np.argsort(importances)[-15:]

plt.figure(figsize=(8,6))
plt.barh(range(len(indices)), importances[indices], align="center")
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.title("Top 15 Feature Importances (XGBoost)")
plt.show()



# ==============================
# 7. Best Model (RandomForest) - Evaluation
# ==============================
best_model = RandomForestClassifier(
    n_estimators=500,
    max_depth=6,
    min_samples_split=50,
    min_samples_leaf=20,
    max_features=0.3,
    class_weight="balanced",
    random_state=42
)

best_model.fit(X_resampled, y_resampled)

RocCurveDisplay.from_estimator(best_model, X_test_final, y_test)
plt.title("ROC Curve"); plt.show()

PrecisionRecallDisplay.from_estimator(best_model, X_test_final, y_test)
plt.title("Precision-Recall Curve"); plt.show()

ConfusionMatrixDisplay.from_estimator(best_model, X_test_final, y_test)
plt.title("Confusion Matrix"); plt.show()

# ==============================
# 8. Learning Curve
# ==============================
train_sizes, train_scores, val_scores = learning_curve(
    best_model,
    X_resampled,
    y_resampled,
    cv=5,
    scoring='f1',
    train_sizes=np.linspace(0.1, 1.0, 5),
    n_jobs=-1
)

train_mean = train_scores.mean(axis=1)
val_mean = val_scores.mean(axis=1)

plt.plot(train_sizes, train_mean, marker='o', label='Train')
plt.plot(train_sizes, val_mean, marker='o', label='Validation')
plt.xlabel('Training examples')
plt.ylabel('F1 Score')
plt.title('Learning Curve')
plt.legend()
plt.grid(True)
plt.show()
# ==============================
# 9. Validation Curve (max_depth)
# ==============================
param_range = [2, 4, 6, 8, 10, 15, 20, None]

train_scores, val_scores = validation_curve(
    RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"),
    X_resampled, y_resampled,
    param_name="max_depth",
    param_range=param_range,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

plt.plot(param_range, train_scores.mean(axis=1), marker="o", label="Train")
plt.plot(param_range, val_scores.mean(axis=1), marker="o", label="Validation")
plt.xlabel("max_depth")
plt.ylabel("F1 Score")
plt.title("Validation Curve - RandomForest")
plt.legend()
plt.show()

# ==============================
# 11. Save Best Model
# ==============================
joblib.dump(best_model, "best_randomforest_model.pkl")
print("Model saved as best_randomforest_model.pkl")

# ==============================
# 12. Save Pipeline
# ==============================
from sklearn.pipeline import Pipeline

# Pipeline = Preprocessing + Model
final_pipeline = Pipeline([
    ("prep", all_pipeline),
    ("clf", RandomForestClassifier(
        n_estimators=500,
        max_depth=6,
        min_samples_split=50,
        min_samples_leaf=20,
        max_features=0.3,
        class_weight="balanced",
        random_state=42
    ))
])


final_pipeline.fit(X_train, y_train)


y_pred = final_pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

joblib.dump(final_pipeline, "churn_pipeline.pkl")
print("Pipeline saved as churn_pipeline.pkl")




