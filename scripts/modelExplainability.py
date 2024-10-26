import pandas as pd
import numpy as np
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# Load your dataset
fraud_data = pd.read_csv('./data/Fraud_Data.csv')  

# Display the first few rows to understand the data
print(fraud_data.head())
print(fraud_data.info())  

# Convert date-time features to a numerical format if present
# Replace 'date_column_name' with the actual column name if it exists
date_column_name = 'date_column_name'  # Update this if necessary
if date_column_name in fraud_data.columns: 
    fraud_data[date_column_name] = pd.to_datetime(fraud_data[date_column_name])
    fraud_data['year'] = fraud_data[date_column_name].dt.year
    fraud_data['month'] = fraud_data[date_column_name].dt.month
    fraud_data['day'] = fraud_data[date_column_name].dt.day
    fraud_data.drop(columns=[date_column_name], inplace=True) 

# Identify categorical columns and encode them
categorical_columns = fraud_data.select_dtypes(include=['object']).columns.tolist()
for col in categorical_columns:
    le = LabelEncoder()
    fraud_data[col] = le.fit_transform(fraud_data[col])

# Prepare data
X = fraud_data.drop(columns=['class'])  # Ensure 'class' is your target column
y = fraud_data['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
rf_model = RandomForestClassifier(random_state=42)  # Added random_state for reproducibility
rf_model.fit(X_train, y_train)

# Predictions and Evaluation
y_pred = rf_model.predict(X_test)
print("Random Forest Classification Report:\n", classification_report(y_test, y_pred))

# --- SHAP Explanation ---
# Initialize the SHAP explainer
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test)

# Print shapes to debug
print("Shape of SHAP values for the positive class:", shap_values[1].shape)
print("Shape of features for the instance:", X_test.iloc[0].shape)

# Summary Plot
shap.summary_plot(shap_values, X_test)

# Force Plot for a specific instance
shap.initjs()
instance_index = 0  # Change this index to visualize different instances
# Ensure you're using the correct class index for SHAP values
shap.force_plot(explainer.expected_value[1], shap_values[1][instance_index], X_test.iloc[instance_index])

# Dependence Plot for a specific feature
feature_name = X.columns[0]  # Replace with the actual feature you want to analyze
shap.dependence_plot(feature_name, shap_values[1], X_test)  

# --- LIME Explanation ---
# Initialize LIME explainer
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train.values, 
    feature_names=X_train.columns, 
    class_names=['Not Fraud', 'Fraud'], 
    mode='classification'
)

# Pick an instance to explain
instance_to_explain = X_test.iloc[instance_index].values
lime_exp = lime_explainer.explain_instance(instance_to_explain, rf_model.predict_proba, num_features=10)

# LIME Feature Importance Plot
lime_exp.as_pyplot_figure()
plt.title(f"LIME Feature Importance for Instance {instance_index}")
plt.show()
