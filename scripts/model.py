# Import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, precision_score, recall_score, f1_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, SimpleRNN, LSTM

import mlflow
import mlflow.sklearn

# Load datasets (replace 'creditcard.csv' and 'fraud_data.csv' with actual file paths)
credit_card_data = pd.read_csv('./data/creditcard.csv')
fraud_data = pd.read_csv('./data/Fraud_Data.csv')

# Separate features and target for both datasets
X_credit = credit_card_data.drop(columns=['Class'])
y_credit = credit_card_data['Class']

X_fraud = fraud_data.drop(columns=['class'])
y_fraud = fraud_data['class']

# Train-test split (80% train, 20% test)
X_credit_train, X_credit_test, y_credit_train, y_credit_test = train_test_split(X_credit, y_credit, test_size=0.2, random_state=42)
X_fraud_train, X_fraud_test, y_fraud_train, y_fraud_test = train_test_split(X_fraud, y_fraud, test_size=0.2, random_state=42)

# Function to evaluate model
def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, preds)}")
    print(f"Precision: {precision_score(y_test, preds)}")
    print(f"Recall: {recall_score(y_test, preds)}")
    print(f"F1 Score: {f1_score(y_test, preds)}")
    print(f"AUC-ROC: {roc_auc_score(y_test, preds)}")
    print("\nClassification Report:\n", classification_report(y_test, preds))

# Logistic Regression Model
def logistic_regression(X_train, y_train, X_test, y_test):
    print("Logistic Regression")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    return model

# Decision Tree Model
def decision_tree(X_train, y_train, X_test, y_test):
    print("Decision Tree")
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    return model

# Random Forest Model
def random_forest(X_train, y_train, X_test, y_test):
    print("Random Forest")
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    return model

# Gradient Boosting Model
def gradient_boosting(X_train, y_train, X_test, y_test):
    print("Gradient Boosting")
    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    return model

# Multi-Layer Perceptron (MLP) Model
def mlp(X_train, y_train, X_test, y_test):
    print("Multi-Layer Perceptron")
    model = MLPClassifier(max_iter=500)
    model.fit(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    return model

# CNN Model
def cnn(X_train, y_train, X_test, y_test):
    print("Convolutional Neural Network")
    X_train = np.expand_dims(X_train, axis=2)
    X_test = np.expand_dims(X_test, axis=2)
    
    model = Sequential()
    model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
    
    preds = (model.predict(X_test) > 0.5).astype("int32")
    print(f"Accuracy: {accuracy_score(y_test, preds)}")
    return model

# RNN Model
def rnn(X_train, y_train, X_test, y_test):
    print("Recurrent Neural Network")
    X_train = np.expand_dims(X_train, axis=2)
    X_test = np.expand_dims(X_test, axis=2)
    
    model = Sequential()
    model.add(SimpleRNN(100, activation='relu', input_shape=(X_train.shape[1], 1)))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
    
    preds = (model.predict(X_test) > 0.5).astype("int32")
    print(f"Accuracy: {accuracy_score(y_test, preds)}")
    return model

# LSTM Model
def lstm(X_train, y_train, X_test, y_test):
    print("Long Short-Term Memory (LSTM)")
    X_train = np.expand_dims(X_train, axis=2)
    X_test = np.expand_dims(X_test, axis=2)
    
    model = Sequential()
    model.add(LSTM(100, activation='relu', input_shape=(X_train.shape[1], 1)))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
    
    preds = (model.predict(X_test) > 0.5).astype("int32")
    print(f"Accuracy: {accuracy_score(y_test, preds)}")
    return model

# MLOps with MLflow
def log_experiment(model_name, model, X_test, y_test):
    mlflow.start_run()
    
    preds = model.predict(X_test)
    
    # Log model and metrics
    mlflow.log_param("model", model_name)
    mlflow.log_metric("accuracy", accuracy_score(y_test, preds))
    mlflow.log_metric("precision", precision_score(y_test, preds))
    mlflow.log_metric("recall", recall_score(y_test, preds))
    mlflow.log_metric("f1_score", f1_score(y_test, preds))
    mlflow.log_metric("auc_roc", roc_auc_score(y_test, preds))
    
    # Log model artifact
    mlflow.sklearn.log_model(model, f"{model_name}_model")
    
    mlflow.end_run()

# Train and evaluate models on Credit Card data
print("CREDIT CARD DATA MODELING")

# Logistic Regression
lr_model = logistic_regression(X_credit_train, y_credit_train, X_credit_test, y_credit_test)
log_experiment("Logistic Regression", lr_model, X_credit_test, y_credit_test)

# Decision Tree
dt_model = decision_tree(X_credit_train, y_credit_train, X_credit_test, y_credit_test)
log_experiment("Decision Tree", dt_model, X_credit_test, y_credit_test)

# Random Forest
rf_model = random_forest(X_credit_train, y_credit_train, X_credit_test, y_credit_test)
log_experiment("Random Forest", rf_model, X_credit_test, y_credit_test)

# Gradient Boosting
gb_model = gradient_boosting(X_credit_train, y_credit_train, X_credit_test, y_credit_test)
log_experiment("Gradient Boosting", gb_model, X_credit_test, y_credit_test)

# Multi-Layer Perceptron
mlp_model = mlp(X_credit_train, y_credit_train, X_credit_test, y_credit_test)
log_experiment("MLP", mlp_model, X_credit_test, y_credit_test)

# CNN
cnn_model = cnn(X_credit_train, y_credit_train, X_credit_test, y_credit_test)

# RNN
rnn_model = rnn(X_credit_train, y_credit_train, X_credit_test, y_credit_test)

# LSTM
lstm_model = lstm(X_credit_train, y_credit_train, X_credit_test, y_credit_test)

# Similarly, you can apply the models and MLOps to the Fraud_Data dataset.
print("\nFRAUD DATA MODELING")

# Apply similar steps as above for Fraud_Data

