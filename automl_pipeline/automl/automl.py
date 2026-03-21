"""
AutoML - Complete Machine Learning Library
A comprehensive ML library that automates the entire pipeline:
Data → EDA → Preprocessing → Split → Model → Evaluate → Predict
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.svm import SVC, SVR
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                           precision_score, recall_score, f1_score,
                           mean_absolute_error, mean_squared_error, r2_score)
import warnings

warnings.filterwarnings('ignore')

# For Jupyter notebooks
try:
    from IPython.display import display, HTML
    IN_JUPYTER = True
except ImportError:
    IN_JUPYTER = False


class AutoML:
    """
    Complete Automated Machine Learning Pipeline
    """
    
    def __init__(self, df, target_col, test_size=0.2, random_state=42, 
                 problem_type='auto', scale_method='standard'):
        self.df = df.copy()
        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state
        self.problem_type = problem_type
        self.scale_method = scale_method
        
        # Store results
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.scaler = None
        self.label_encoders = {}
        
        # Auto-detect problem type
        self._detect_problem_type()
        
        print("="*80)
        print("🤖 AutoML Initialized")
        print("="*80)
        print(f"📊 Dataset: {self.df.shape[0]} rows × {self.df.shape[1]} columns")
        print(f"🎯 Target: {target_col}")
        print(f"📈 Problem Type: {self.problem_type.upper()}")
        print(f"🧪 Test Size: {test_size*100}%")
        print("="*80)
    
    def _detect_problem_type(self):
        """Auto-detect if it's classification or regression"""
        if self.problem_type != 'auto':
            return
        
        # Check target column
        target_data = self.df[self.target_col]
        
        if target_data.dtype == 'object' or target_data.nunique() < 10:
            self.problem_type = 'classification'
        else:
            self.problem_type = 'regression'
    
    def eda(self):
        """Perform comprehensive Exploratory Data Analysis"""
        print("\n" + "="*80)
        print("📊 EXPLORATORY DATA ANALYSIS (EDA)")
        print("="*80)
        
        print("\n1️⃣ DATASET OVERVIEW")
        print("-"*40)
        print(f"   Shape: {self.df.shape}")
        print(f"   Columns: {list(self.df.columns)}")
        print(f"   Memory Usage: {self.df.memory_usage().sum() / 1024**2:.2f} MB")
        
        print("\n2️⃣ FIRST 5 ROWS")
        print("-"*40)
        print(self.df.head())
        
        print("\n3️⃣ LAST 5 ROWS")
        print("-"*40)
        print(self.df.tail())
        
        print("\n4️⃣ DATA TYPES")
        print("-"*40)
        dtype_df = pd.DataFrame(self.df.dtypes).reset_index()
        dtype_df.columns = ['Column', 'Type']
        print(dtype_df)
        
        print("\n5️⃣ MISSING VALUES")
        print("-"*40)
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        missing_df = pd.DataFrame({
            'Missing': missing,
            'Percentage': missing_pct
        })
        print(missing_df[missing_df['Missing'] > 0] if missing.sum() > 0 else "No missing values found")
        
        print("\n6️⃣ DUPLICATES")
        print("-"*40)
        print(f"   Duplicate rows: {self.df.duplicated().sum()}")
        
        print("\n7️⃣ STATISTICAL SUMMARY")
        print("-"*40)
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            print(self.df[numeric_cols].describe())
        else:
            print("No numeric columns found")
        
        cat_cols = self.df.select_dtypes(include=['object']).columns
        if len(cat_cols) > 0:
            print("\n8️⃣ CATEGORICAL COLUMNS")
            print("-"*40)
            for col in cat_cols:
                print(f"\n   {col}:")
                print(f"   Unique values: {self.df[col].nunique()}")
                print(f"   Top values:\n{self.df[col].value_counts().head()}")
        
        print("\n9️⃣ TARGET DISTRIBUTION")
        print("-"*40)
        if self.problem_type == 'classification':
            print(self.df[self.target_col].value_counts())
            if IN_JUPYTER:
                self.df[self.target_col].value_counts().plot(kind='bar', title='Target Distribution')
                plt.show()
        else:
            print(self.df[self.target_col].describe())
            if IN_JUPYTER:
                self.df[self.target_col].hist(bins=30, title='Target Distribution')
                plt.show()
        
        if len(numeric_cols) > 1:
            print("\n🔟 CORRELATION MATRIX")
            print("-"*40)
            corr = self.df[numeric_cols].corr()
            print(corr)
            
            if self.target_col in numeric_cols:
                target_corr = corr[self.target_col].sort_values(ascending=False)
                print(f"\n   Correlations with {self.target_col}:")
                print(target_corr)
        
        print("\n" + "="*80)
        print("✅ EDA COMPLETED")
        print("="*80)
        
        return self
    
    def preprocess(self, handle_missing=True, encode_categorical=True, 
                   scale_features=True, create_polynomial=False, poly_degree=2):
        print("\n" + "="*80)
        print("🛠️  DATA PREPROCESSING")
        print("="*80)
        
        df_processed = self.df.copy()
        
        if handle_missing:
            print("\n1️⃣ HANDLING MISSING VALUES")
            print("-"*40)
            for col in df_processed.columns:
                if df_processed[col].isnull().sum() > 0:
                    if df_processed[col].dtype == 'object':
                        mode_val = df_processed[col].mode()[0]
                        df_processed[col].fillna(mode_val, inplace=True)
                        print(f"   ✓ {col}: Filled with mode ({mode_val})")
                    else:
                        median_val = df_processed[col].median()
                        df_processed[col].fillna(median_val, inplace=True)
                        print(f"   ✓ {col}: Filled with median ({median_val:.2f})")
        
        if encode_categorical:
            print("\n2️⃣ ENCODING CATEGORICAL VARIABLES")
            print("-"*40)
            cat_cols = df_processed.select_dtypes(include=['object']).columns
            for col in cat_cols:
                if col != self.target_col or self.problem_type == 'classification':
                    le = LabelEncoder()
                    df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                    self.label_encoders[col] = le
                    print(f"   ✓ {col}: Label Encoded")
        
        self.X = df_processed.drop(self.target_col, axis=1)
        self.y = df_processed[self.target_col]
        
        if create_polynomial and self.problem_type == 'regression':
            print("\n3️⃣ CREATING POLYNOMIAL FEATURES")
            print("-"*40)
            poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
            self.X = pd.DataFrame(
                poly.fit_transform(self.X),
                columns=poly.get_feature_names_out(self.X.columns)
            )
            print(f"   ✓ Created {self.X.shape[1]} polynomial features")
        
        if scale_features and self.scale_method:
            print("\n4️⃣ SCALING FEATURES")
            print("-"*40)
            if self.scale_method == 'standard':
                self.scaler = StandardScaler()
            elif self.scale_method == 'minmax':
                self.scaler = MinMaxScaler()
            
            self.X_scaled = self.scaler.fit_transform(self.X)
            self.X = pd.DataFrame(self.X_scaled, columns=self.X.columns)
            print(f"   ✓ Applied {self.scale_method.upper()} scaling")
        
        print(f"\n✅ Final Feature Shape: {self.X.shape}")
        print(f"✅ Target Shape: {self.y.shape}")
        print("="*80)
        
        return self
    
    def split_data(self):
        print("\n" + "="*80)
        print("📂 TRAIN-TEST SPLIT")
        print("="*80)
        
        stratify = self.y if self.problem_type == 'classification' else None
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, 
            random_state=self.random_state, stratify=stratify
        )
        
        print(f"📊 Training set: {self.X_train.shape[0]} samples")
        print(f"📊 Testing set: {self.X_test.shape[0]} samples")
        
        if self.problem_type == 'classification':
            print(f"\n   Training class distribution:")
            print(f"   {pd.Series(self.y_train).value_counts().to_dict()}")
            print(f"\n   Testing class distribution:")
            print(f"   {pd.Series(self.y_test).value_counts().to_dict()}")
        
        print("="*80)
        return self
    
    def _get_classification_models(self):
        return {
            'KNN (k=3)': KNeighborsClassifier(n_neighbors=3),
            'KNN (k=5)': KNeighborsClassifier(n_neighbors=5),
            'KNN (k=7)': KNeighborsClassifier(n_neighbors=7),
            'Naive Bayes': GaussianNB(),
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=self.random_state),
            'Decision Tree': DecisionTreeClassifier(random_state=self.random_state),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=self.random_state),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=self.random_state),
            'SVM': SVC(kernel='rbf', random_state=self.random_state, probability=True)
        }
    
    def _get_regression_models(self):
        return {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=1.0),
            'KNN (k=3)': KNeighborsRegressor(n_neighbors=3),
            'KNN (k=5)': KNeighborsRegressor(n_neighbors=5),
            'KNN (k=7)': KNeighborsRegressor(n_neighbors=7),
            'Decision Tree': DecisionTreeRegressor(random_state=self.random_state),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=self.random_state),
            'SVR': SVR(kernel='rbf')
        }
    
    def train_models(self):
        print("\n" + "="*80)
        print(f"🤖 TRAINING {self.problem_type.upper()} MODELS")
        print("="*80)
        
        if self.problem_type == 'classification':
            models = self._get_classification_models()
        else:
            models = self._get_regression_models()
        
        for name, model in models.items():
            try:
                print(f"\n📈 Training {name}...")
                model.fit(self.X_train, self.y_train)
                self.models[name] = model
                print(f"   ✅ {name} trained successfully")
            except Exception as e:
                print(f"   ❌ {name} failed: {str(e)}")
        
        print("\n" + "="*80)
        print(f"✅ Trained {len(self.models)} models")
        print("="*80)
        
        return self
    
    def evaluate_all(self):
        print("\n" + "="*80)
        print(f"📊 EVALUATING {self.problem_type.upper()} MODELS")
        print("="*80)
        
        self.results = {}
        best_score = -np.inf
        
        for name, model in self.models.items():
            print(f"\n{'='*60}")
            print(f"🔍 {name}")
            print("="*60)
            
            y_pred = model.predict(self.X_test)
            
            if self.problem_type == 'classification':
                accuracy = accuracy_score(self.y_test, y_pred)
                precision = precision_score(self.y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(self.y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(self.y_test, y_pred, average='weighted', zero_division=0)
                
                print(f"\n📈 ACCURACY: {accuracy:.4f} ({accuracy*100:.2f}%)")
                print(f"🎯 PRECISION: {precision:.4f}")
                print(f"📌 RECALL: {recall:.4f}")
                print(f"⚡ F1-SCORE: {f1:.4f}")
                
                cm = confusion_matrix(self.y_test, y_pred)
                print(f"\n📊 CONFUSION MATRIX:")
                print(cm)
                
                print(f"\n📋 CLASSIFICATION REPORT:")
                print(classification_report(self.y_test, y_pred, zero_division=0))
                
                try:
                    cv_scores = cross_val_score(model, self.X, self.y, cv=5)
                    print(f"\n🔄 CROSS-VALIDATION (5-fold):")
                    print(f"   Mean: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
                except:
                    pass
                
                self.results[name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                }
                
                if accuracy > best_score:
                    best_score = accuracy
                    self.best_model = model
                    self.best_model_name = name
                    
            else:
                r2 = r2_score(self.y_test, y_pred)
                mae = mean_absolute_error(self.y_test, y_pred)
                mse = mean_squared_error(self.y_test, y_pred)
                rmse = np.sqrt(mse)
                
                print(f"\n📈 R² SCORE: {r2:.4f}")
                print(f"📊 MAE: {mae:.4f}")
                print(f"📉 MSE: {mse:.4f}")
                print(f"📏 RMSE: {rmse:.4f}")
                
                try:
                    cv_scores = cross_val_score(model, self.X, self.y, cv=5, scoring='r2')
                    print(f"\n🔄 CROSS-VALIDATION (5-fold):")
                    print(f"   Mean R²: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
                except:
                    pass
                
                self.results[name] = {
                    'r2': r2,
                    'mae': mae,
                    'mse': mse,
                    'rmse': rmse
                }
                
                if r2 > best_score:
                    best_score = r2
                    self.best_model = model
                    self.best_model_name = name
        
        return self
    
    def show_best_model(self):
        print("\n" + "="*80)
        print("🏆 BEST MODEL")
        print("="*80)
        print(f"\n📌 Model: {self.best_model_name}")
        
        if hasattr(self.best_model, 'feature_importances_'):
            print(f"\n📈 Feature Importance:")
            importances = pd.DataFrame({
                'Feature': self.X.columns,
                'Importance': self.best_model.feature_importances_
            }).sort_values('Importance', ascending=False)
            print(importances.head(10))
        
        return self
    
    def predict_new(self, new_data):
        if self.best_model is None:
            print("❌ No model trained yet. Run train_models() first.")
            return None
        
        print("\n" + "="*80)
        print("🔮 PREDICTIONS")
        print("="*80)
        
        if isinstance(new_data, (list, np.ndarray)):
            new_df = pd.DataFrame([new_data], columns=self.X.columns if hasattr(self, 'X') and self.X is not None else None)
        else:
            new_df = new_data.copy()
        
        if self.scaler:
            new_df = pd.DataFrame(
                self.scaler.transform(new_df),
                columns=new_df.columns
            )
        
        predictions = self.best_model.predict(new_df)
        
        print(f"\n📌 Predictions:")
        for i, pred in enumerate(predictions):
            print(f"   Sample {i+1}: {pred}")
        
        if self.problem_type == 'classification' and hasattr(self.best_model, 'predict_proba'):
            probs = self.best_model.predict_proba(new_df)
            print(f"\n📊 Probabilities:")
            for i, prob in enumerate(probs):
                print(f"   Sample {i+1}: {prob}")
        
        return predictions
    
    def compare_models(self):
        print("\n" + "="*80)
        print("📊 MODEL COMPARISON")
        print("="*80)
        
        if self.results:
            comparison_df = pd.DataFrame(self.results).T
            print(comparison_df)
            
            best_metric = 'accuracy' if self.problem_type == 'classification' else 'r2'
            best_value = comparison_df[best_metric].max()
            print(f"\n🏆 Best model by {best_metric.upper()}: {best_value:.4f}")
            
            return comparison_df
        
        print("No results available. Run evaluate_all() first.")
        return None
    
    def run_complete_analysis(self):
        print("\n" + "🚀"*40)
        print("COMPLETE AUTOMATED ML ANALYSIS")
        print("🚀"*40)
        
        self.eda()
        self.preprocess()
        self.split_data()
        self.train_models()
        self.evaluate_all()
        self.show_best_model()
        
        print("\n" + "✅"*40)
        print("ANALYSIS COMPLETE!")
        print("✅"*40)
        
        return self


def quick_ml(df, target, problem_type='auto', test_size=0.2):
    ml = AutoML(df, target, problem_type=problem_type, test_size=test_size)
    ml.run_complete_analysis()
    return ml


def compare_algorithms(df, target, problem_type='auto'):
    ml = AutoML(df, target, problem_type=problem_type)
    ml.preprocess()
    ml.split_data()
    ml.train_models()
    ml.evaluate_all()
    return ml.compare_models()
