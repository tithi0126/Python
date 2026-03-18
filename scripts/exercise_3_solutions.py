import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler

def evaluate_model(y_test, y_pred, model_name):
    print(f"\n--- Performance of {model_name} ---")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, average='weighted', zero_division=0))
    print("Recall:", recall_score(y_test, y_pred, average='weighted', zero_division=0))

# 1. Naïve Bayesian Classifier (pima_indian.csv)
def task_1_naive_bayes():
    print("\n--- Task 1: Naive Bayes (pima_indian.csv) ---")
    df = pd.read_csv('pima_indian.csv')
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    evaluate_model(y_test, y_pred, "Naive Bayes")

# 2. Bayesian Network (heart.csv) - Simplified Probabilistic Model
# Since pgmpy is not available, we use Naive Bayes as the underlying probabilistic model
def task_2_bayesian_network():
    print("\n--- Task 2: Bayesian Context (heart.csv) ---")
    df = pd.read_csv('heart.csv')
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = GaussianNB() # Using GNB to approximate Bayesian inference
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    evaluate_model(y_test, y_pred, "Bayesian Model")

# 3. kNN Classifier (titanic.csv)
def task_3_knn_titanic():
    print("\n--- Task 3: kNN (titanic.csv) ---")
    df = pd.read_csv('titanic.csv')
    # Preprocessing
    df = df[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
    df['Age'] = df['Age'].fillna(df['Age'].mean())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    le = LabelEncoder()
    df['Sex'] = le.fit_transform(df['Sex'])
    df['Embarked'] = le.fit_transform(df['Embarked'])
    
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    evaluate_model(y_test, y_pred, "kNN Titanic")

# 4. kNN Classifier (winequalityN.csv)
def task_4_knn_wine():
    print("\n--- Task 4: kNN (winequalityN.csv) ---")
    df = pd.read_csv('winequalityN.csv')
    df = df.dropna()
    le = LabelEncoder()
    df['type'] = le.fit_transform(df['type'])
    X = df.drop('quality', axis=1)
    y = df['quality']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    evaluate_model(y_test, y_pred, "kNN Wine Quality")

# 5. Algorithm Comparison (customer_churn.csv)
def task_5_comparison():
    print("\n--- Task 5: NB vs kNN (customer_churn.csv) ---")
    df = pd.read_csv('customer_churn.csv')
    # Simplified preprocessing for categorical data
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = LabelEncoder().fit_transform(df[col])
    
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # NB
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    y_pred_nb = nb.predict(X_test)
    evaluate_model(y_test, y_pred_nb, "Naive Bayes Churn")
    
    # kNN
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_s, y_train)
    y_pred_knn = knn.predict(X_test_s)
    evaluate_model(y_test, y_pred_knn, "kNN Churn")

# 6. Performance & Prediction (winequalityN.csv)
def task_6_prediction():
    print("\n--- Task 6: Custom Prediction (winequalityN.csv) ---")
    df = pd.read_csv('winequalityN.csv').dropna()
    le = LabelEncoder()
    df['type'] = le.fit_transform(df['type'])
    X = df.drop('quality', axis=1)
    y = df['quality']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train_s, y_train)
    
    print("\n[Predicting for User Input (Sample)]")
    sample_input = X.iloc[0].values.reshape(1, -1)
    sample_input_s = scaler.transform(sample_input)
    prediction = model.predict(sample_input_s)
    print("Predicted Quality for Sample 0:", prediction[0])
    print("Actual Quality for Sample 0:", y.iloc[0])

if __name__ == "__main__":
    task_1_naive_bayes()
    task_2_bayesian_network()
    task_3_knn_titanic()
    task_4_knn_wine()
    task_5_comparison()
    task_6_prediction()
