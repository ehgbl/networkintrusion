#!/usr/bin/env python3
"""
Simple Example: Network Intrusion Detection System
This script shows how to use the system
"""

from network_intrusion_detection import NetworkIntrusionDetector
import pandas as pd

def simple_example():
    """Simple example of using the Network Intrusion Detection System"""
    
    print("Simple Network Intrusion Detection Example")
    print("="*50)
    
    print("Initializing the detector...")
    detector = NetworkIntrusionDetector()
    
    print("\nLoading your data...")
    data = detector.load_and_preprocess_data('Test_data.csv')
    
    print("\nPreparing features...")
    features, target = detector.prepare_features()
    
    print("\nSplitting data...")
    X_train_scaled, X_test_scaled, y_train, y_test = detector.split_data()
    
    print("\nTraining Random Forest...")
    rf_model, rf_accuracy = detector.train_random_forest()
    
    print("\nTraining SVM...")
    svm_model, svm_accuracy = detector.train_svm()
    
    print("\nComparing results...")
    print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
    print(f"SVM Accuracy: {svm_accuracy:.4f}")
    
    if rf_accuracy > svm_accuracy:
        print("Random Forest is better for your data!")
    elif svm_accuracy > rf_accuracy:
        print("SVM is better for your data!")
    else:
        print("Both models perform equally!")
    
    print("\nExample completed! Check the generated visualization files.")

if __name__ == "__main__":
    simple_example()
