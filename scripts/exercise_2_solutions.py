import numpy as np
import pandas as pd

# 1. Arithmetic Function
def calc(a, b, op):
    if op == 1:
        return a + b
    elif op == 2:
        return a - b
    elif op == 3:
        return a * b
    elif op == 4:
        if b != 0:
            return a / b
        else:
            print("Error: Division by zero")
            return None
    else:
        print("Error: Invalid operator")
        return None

# 2. Word Length Helper
def word_length_helper(words):
    if not words:
        return None
    shortest = min(words, key=len)
    longest = max(words, key=len)
    return (shortest, longest)

# 3. Unique List Adder
def unique_list_adder(L, element):
    if element not in L:
        L.append(element)
    return L

# 4. Numpy Transpose
def numpy_transpose(arr):
    return np.transpose(arr)

# 5. Numpy Row Vector
def list_to_row_vector(L):
    return np.array(L).reshape(1, -1)

# 6. Numpy Column Vector
def extract_last_column_vector(arr):
    return arr[:, -1].reshape(-1, 1)

# 7. Pandas EDA on 'adult data.csv'
def perform_eda():
    print("\n--- Pandas EDA on adult data.csv ---")
    df = pd.read_csv('adult data.csv')
    
    # 1. head()
    print("1. Head:\n", df.head())
    
    # 2. shape
    print("\n2. Shape:", df.shape)
    
    # 3. columns
    print("\n3. Columns:", df.columns)
    
    # 4. info()
    print("\n4. Info:")
    df.info()
    
    # 5. describe()
    print("\n5. Descriptive Statistics:\n", df.describe())
    
    # 6. isnull().sum()
    print("\n6. Missing Values:\n", df.isnull().sum())
    
    # 7. value_counts() for a categorical column
    # Let's check first column index 14 which is usually the target in adult census
    target_col = df.columns[-1]
    print(f"\n7. Value counts for {target_col}:\n", df[target_col].value_counts())
    
    # 8. Data Preprocessing: Strip whitespaces if any (common in adult dataset)
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    print("\n8. Stripped whitespaces from object columns.")
    
    # 9. drop_duplicates()
    df = df.drop_duplicates()
    print("\n9. Shape after dropping duplicates:", df.shape)
    
    # 10. unique()
    print("\n10. Unique values in first column:", df[df.columns[0]].unique()[:10])

if __name__ == "__main__":
    # Test Functions
    print("Calc(10, 5, 1):", calc(10, 5, 1))
    print("Word Helper:", word_length_helper(["apple", "pie", "banana"]))
    print("Unique Adder:", unique_list_adder([1, 2, 3], 4))
    
    # Test Numpy
    arr = np.array([[1, 2], [3, 4]])
    print("Transpose:\n", numpy_transpose(arr))
    print("Row Vector:", list_to_row_vector([1, 2, 3]))
    print("Column Vector:\n", extract_last_column_vector(arr))
    
    # Test EDA
    perform_eda()
