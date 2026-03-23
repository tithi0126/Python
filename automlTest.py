import os
import pandas as pd

# If using PIP package, importing directly
try:
    from automl import AutoML
except ImportError:
    import sys
    sys.path.insert(0, './automl_pipeline/automl')
    from automl import AutoML

def main():
    datasets_dir = './datasets'
    
    print("="*80)
    print("🚀 BATCH RUNNING AutoML ON COMPLETELY UNKNOWN DATASETS 🚀")
    print("="*80)
    
    if not os.path.exists(datasets_dir):
        print(f"Error: {datasets_dir} folder not found.")
        return
        
    csv_files = [f for f in os.listdir(datasets_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print("No CSV files found in the datasets directory.")
        return

    # To keep track of overall results
    all_best_models = {}

    # Iterate through all CSV files blindly
    for file in csv_files:
        file_path = os.path.join(datasets_dir, file)
        
        # When dataset AND target are unknown, we ALWAYS assume the LAST COLUMN is the target variable.
        try:
            temp_df = pd.read_csv(file_path, nrows=1)
            target_col = temp_df.columns[-1]
            if "Unnamed" in target_col:
                target_col = temp_df.columns[-2] # fallback if trailing commas exist
        except:
            print(f"❌ ERROR: Could not read {file}. Skipping.")
            continue
        
        print(f"\n\n{'*'*80}")
        print(f"🌟 PROCESSING COMPLETELY UNKNOWN DATASET: {file}")
        print(f"🎯 Auto-Detected Target Column: '{target_col}'")
        print(f"{'*'*80}\n")
        
        try:
            df = pd.read_csv(file_path)
            
            # Create and run AutoML
            ml = AutoML(
                df=df,
                target_col=target_col,
                test_size=0.2,
                problem_type='auto' # Automatically detects classification vs regression
            )
            
            # Run full analysis
            ml.run_complete_analysis()
            
            # Print and save Best Model Info
            print(f"\n✅ {file} COMPLETED SUCCESSFULLY")
            if hasattr(ml, 'best_model_name') and ml.best_model_name:
                print(f"🏆 Best Model Found: {ml.best_model_name}")
                print(f"📈 Best Score: {ml.results[ml.best_model_name]}")
                all_best_models[file] = ml.best_model_name
            else:
                print("⚠️ No models successfully trained for this dataset.")
            
        except Exception as e:
            import traceback
            print(f"\n❌ ERROR running analysis for {file}:")
            traceback.print_exc()

    # Final summary of all datasets
    print("\n\n" + "="*80)
    print("📊 FINAL BATCH RESULTS FOR UNKNOWN DATASETS 📊")
    print("="*80)
    for dataset, best_model in all_best_models.items():
        print(f"✔️ {dataset.ljust(25)} -> 🏆 Best Model: {best_model}")

if __name__ == "__main__":
    main()
