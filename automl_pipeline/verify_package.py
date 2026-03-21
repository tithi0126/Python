import pandas as pd
import numpy as np
from automl import AutoML

def test_pipeline():
    print("🚀 Running AutoML Verification...")
    
    # Create sample dataset
    np.random.seed(42)
    n_samples = 100
    df = pd.DataFrame({
        'feature1': np.random.rand(n_samples),
        'feature2': np.random.rand(n_samples),
        'target': np.random.choice([0, 1], n_samples)
    })
    
    # Initialize AutoML
    ml = AutoML(df, 'target', problem_type='classification')
    
    # Run analysis
    ml.preprocess()
    ml.split_data()
    ml.train_models()
    ml.evaluate_all()
    
    print("\n✅ Verification Successful!")

if __name__ == "__main__":
    test_pipeline()
