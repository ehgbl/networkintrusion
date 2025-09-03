#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class NetworkIntrusionDetector:
    def __init__(self):
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.svm_model = SVC(kernel='rbf', random_state=42)
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        
    def load_and_preprocess_data(self, file_path):
        print("Loading data...")
        self.data = pd.read_csv(file_path)
        print(f"Dataset shape: {self.data.shape}")
        print(f"Columns: {list(self.data.columns)}")
        
        print("\nDataset Info:")
        print(self.data.info())
        
        print("\nMissing values:")
        print(self.data.isnull().sum())
        
        print("\nFirst 5 rows:")
        print(self.data.head())
        
        # TODO: add data validation here
        # if self.data.empty:
        #     raise ValueError("Dataset is empty!")
        
        return self.data
    
    def prepare_features(self):
        print("\nPreparing features...")
        
        if 'flag' in self.data.columns:
            self.features = self.data.drop('flag', axis=1)
            self.target = self.data['flag']
        else:
            print("No target column found. Creating synthetic labels for demonstration...")
            self.features = self.data
            np.random.seed(42)
            self.target = np.random.choice([0, 1], size=len(self.data), p=[0.7, 0.3])
        
        categorical_columns = self.features.select_dtypes(include=['object']).columns
        
        for col in categorical_columns:
            if col in self.features.columns:
                le = LabelEncoder()
                self.features[col] = le.fit_transform(self.features[col].astype(str))
                self.label_encoders[col] = le
        
        self.features = self.features.astype(float)
        
        self.feature_names = list(self.features.columns)
        
        print(f"Features shape: {self.features.shape}")
        print(f"Target shape: {self.target.shape}")
        print(f"Target distribution:\n{self.target.value_counts()}")
        
        # Debug: check for infinite values
        # if np.isinf(self.features.values).any():
        #     print("Warning: Infinite values detected in features!")
        
        return self.features, self.target
    
    def split_data(self, test_size=0.2):
        print("\nSplitting data...")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.features, self.target, test_size=test_size, random_state=42, stratify=self.target
        )
        
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Training set size: {self.X_train.shape[0]}")
        print(f"Testing set size: {self.X_test.shape[0]}")
        
        # Check for scaling issues
        # if np.isnan(self.X_train_scaled).any():
        #     print("Warning: NaN values after scaling!")
        
        return self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test
    
    def train_random_forest(self):
        print("\nTraining Random Forest model...")
        
        self.rf_model.fit(self.X_train_scaled, self.y_train)
        
        self.rf_predictions = self.rf_model.predict(self.X_test_scaled)
        self.rf_accuracy = accuracy_score(self.y_test, self.rf_predictions)
        
        self.rf_cv_scores = cross_val_score(self.rf_model, self.X_train_scaled, self.y_train, cv=5)
        
        print(f"Random Forest Accuracy: {self.rf_accuracy:.4f}")
        print(f"Cross-validation scores: {self.rf_cv_scores}")
        print(f"Mean CV accuracy: {self.rf_cv_scores.mean():.4f} (+/- {self.rf_cv_scores.std() * 2:.4f})")
        
        # Performance check
        # if self.rf_accuracy < 0.5:
        #     print("Warning: Random Forest accuracy is very low!")
        
        return self.rf_model, self.rf_accuracy
    
    def train_svm(self):
        print("\nTraining SVM model...")
        
        self.svm_model.fit(self.X_train_scaled, self.y_train)
        
        self.svm_predictions = self.svm_model.predict(self.X_test_scaled)
        self.svm_accuracy = accuracy_score(self.y_test, self.svm_predictions)
        
        self.svm_cv_scores = cross_val_score(self.svm_model, self.X_train_scaled, self.y_train, cv=5)
        
        print(f"SVM Accuracy: {self.svm_accuracy:.4f}")
        print(f"Cross-validation scores: {self.svm_cv_scores}")
        print(f"Mean CV accuracy: {self.svm_cv_scores.mean():.4f} (+/- {self.svm_cv_scores.std() * 2:.4f})")
        
        return self.svm_model, self.svm_accuracy
    
    def evaluate_models(self):
        print("\n" + "="*50)
        print("MODEL EVALUATION RESULTS")
        print("="*50)
        
        print("\nRANDOM FOREST RESULTS:")
        print(f"Accuracy: {self.rf_accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(self.y_test, self.rf_predictions))
        
        print("\nSVM RESULTS:")
        print(f"Accuracy: {self.svm_accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(self.y_test, self.svm_predictions))
        
        print("\nMODEL COMPARISON:")
        print(f"Random Forest: {self.rf_accuracy:.4f}")
        print(f"SVM: {self.svm_accuracy:.4f}")
        
        if self.rf_accuracy > self.svm_accuracy:
            print("Random Forest performs better!")
        elif self.svm_accuracy > self.rf_accuracy:
            print("SVM performs better!")
        else:
            print("Both models perform equally!")
        
        # Log results for later analysis
        # with open('model_results.txt', 'a') as f:
        #     f.write(f"{pd.Timestamp.now()}: RF={self.rf_accuracy:.4f}, SVM={self.svm_accuracy:.4f}\n")
        
        return self.rf_accuracy, self.svm_accuracy
    
    def feature_importance_analysis(self):
        print("\nFEATURE IMPORTANCE ANALYSIS (Random Forest):")
        
        importance = self.rf_model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(feature_importance_df.head(10))
        
        plt.figure(figsize=(12, 8))
        top_features = feature_importance_df.head(15)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title('Top 15 Most Important Features (Random Forest)')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save feature importance to CSV for reference
        # feature_importance_df.to_csv('feature_importance.csv', index=False)
        
        return feature_importance_df
    
    def confusion_matrix_analysis(self):
        print("\nCreating confusion matrices...")
        
        plt.figure(figsize=(15, 6))
        
        plt.subplot(1, 2, 1)
        cm_rf = confusion_matrix(self.y_test, self.rf_predictions)
        sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Normal', 'Attack'], 
                    yticklabels=['Normal', 'Attack'])
        plt.title('Random Forest Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        plt.subplot(1, 2, 2)
        cm_svm = confusion_matrix(self.y_test, self.svm_predictions)
        sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Reds', 
                    xticklabels=['Normal', 'Attack'], 
                    yticklabels=['Normal', 'Attack'])
        plt.title('SVM Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_complete_analysis(self, file_path):
        print("NETWORK INTRUSION DETECTION SYSTEM")
        print("="*50)
        
        self.load_and_preprocess_data(file_path)
        
        self.prepare_features()
        
        self.split_data()
        
        self.train_random_forest()
        self.train_svm()
        
        self.evaluate_models()
        
        self.feature_importance_analysis()
        
        self.confusion_matrix_analysis()
        
        print("\n" + "="*50)
        print("ANALYSIS COMPLETE!")
        print("="*50)
        print("Generated files:")
        print("- feature_importance.png")
        print("- confusion_matrices.png")

def main():
    detector = NetworkIntrusionDetector()
    
    # Add some error handling
    # try:
    #     detector.run_complete_analysis('Test_data.csv')
    # except Exception as e:
    #     print(f"Error during analysis: {e}")
    #     # Log the error
    #     import traceback
    #     traceback.print_exc()
    
    detector.run_complete_analysis('Test_data.csv')

if __name__ == "__main__":
    main()
