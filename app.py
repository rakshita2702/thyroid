# -*- coding: utf-8 -*-
"""newprojectCI01

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1odCnMUSGkhaU18rOMJdjmUwtDu9QThDu
"""

pip install catboost

pip install lightgbm

pip install --upgrade scikit-learn

pip install umap-learn

import time
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import xgboost as xgb
from catboost import CatBoostClassifier
import lightgbm as lgb

#Data Loading and Preprocessing

csv_file_path = "/content/hypothyroid.csv"
df = pd.read_csv(csv_file_path)

# Remove rows containing "?" values
df = df[(df != "?").all(axis=1)]

# Map binary values
# Replace ... with the actual binary column names
binary_cols = ['on thyroxine', 'query on thyroxine', 'on antithyroid medication', 'sick', 'pregnant', 'thyroid surgery', 'I131 treatment', 'query hypothyroid', 'query hyperthyroid', 'lithium', 'goitre', 'tumor', 'hypopituitary', 'psych', 'TSH measured', 'T3 measured', 'TT4 measured', 'T4U measured', 'FTI measured', 'TBG measured']
for col in binary_cols:
    df[col] = df[col].map({'t': 1, 'f': 0})

# Encode categorical columns
df['sex'] = df['sex'].map({'M': 1, 'F': 0})
df['binaryClass'] = df['binaryClass'].map({'P': 1, 'N': 0})
df['referral source'] = LabelEncoder().fit_transform(df['referral source'])

# Split data into features and target
X = df.drop('binaryClass', axis=1)
y = df['binaryClass']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle missing values
imputer = SimpleImputer(strategy='most_frequent')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Exploratory Data Analysis (EDA)

plt.figure(figsize=(6, 4))
sns.countplot(x='binaryClass', data=df)
plt.title('Thyroid Prediction')
plt.show()

plt.figure(figsize=(6, 6))
df['binaryClass'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Thyroid Prediction')
plt.show()

num_rows = int(len(binary_cols) / 3) + (len(binary_cols) % 3 > 0) # Calculate num_rows based on the number of binary_cols

fig, axes = plt.subplots(nrows=num_rows, ncols=3, figsize=(18, num_rows * 5))

for i, col in enumerate(binary_cols):

    # Check if the current index 'i' is within the bounds of the axes array
    if i < len(axes.flatten()):
        category_counts = df[col].value_counts()
        axes.flatten()[i].pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', startangle=140)
        axes.flatten()[i].set_title(f'Pie Chart of {col}')


plt.tight_layout()  # Adjust layout to prevent overlapping
plt.show()

#Model Training and Evaluation

model_xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
model_xgb.fit(X_train, y_train)
y_pred = model_xgb.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

def plot_feature_importance(model, feature_names):
    ...
    plt.barh(feature_importance.index, feature_importance)
    plt.title('Feature Importance')
    plt.show()

model_mlp = MLPClassifier(random_state=42)
model_mlp.fit(X_train, y_train)
y_pred_mlp = model_mlp.predict(X_test)

model_dt = DecisionTreeClassifier(random_state=42)
model_dt.fit(X_train, y_train)
y_pred_dt = model_dt.predict(X_test)

model_svm = SVC(random_state=42)
model_svm.fit(X_train, y_train)
y_pred_svm = model_svm.predict(X_test)

model_rf = RandomForestClassifier(random_state=42)
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)

model_gb = GradientBoostingClassifier(random_state=42)
model_gb.fit(X_train, y_train)
y_pred_gb = model_gb.predict(X_test)

#Model Comparison with PCA

!pip install scikit-learn
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

model_xgb_pca = XGBClassifier(random_state=42)
model_xgb_pca.fit(X_train_pca, y_train)

cv_scores_train = cross_val_score(model_xgb, X_train, y_train, cv=5, scoring='accuracy')
cv_scores_test = cross_val_score(model_xgb, X_test, y_test, cv=5, scoring='accuracy')

#Hyperparameter Tuning
!pip install scikit-learn
from sklearn.model_selection import GridSearchCV # Importing the GridSearchCV class
from sklearn.model_selection import cross_val_score

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3]
}
grid_search = GridSearchCV(estimator=model_xgb, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

!pip install matplotlib scikit-learn xgboost
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
# Assuming X_test, y_test, model_xgb, and model_xgb_pca are defined and available

# Calculate accuracy for model_xgb
y_pred_xgb = model_xgb.predict(X_test)
accuracy_test_xgb = accuracy_score(y_test, y_pred_xgb)

# Calculate accuracy for model_xgb_pca
y_pred_xgb_pca = model_xgb_pca.predict(X_test_pca)
accuracy_test_xgb_pca = accuracy_score(y_test, y_pred_xgb_pca)

# Now you can plot the accuracies
plt.bar(['Without PCA', 'With PCA'], [accuracy_test_xgb, accuracy_test_xgb_pca])
plt.ylabel('Accuracy')
plt.title('Model Comparison with/without PCA')
plt.show()

probs = model_xgb.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, probs)
plt.plot(fpr, tpr, label='XGBoost')

#Saving the Model

import joblib
joblib.dump(model_xgb, 'xgb_model.pkl')