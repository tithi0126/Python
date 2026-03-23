# arti

A complete automated Machine Learning pipeline package created by arti.

## Usage

```python
import pandas as pd
from arti import AutoML

df = pd.read_csv('your_dataset.csv')
ml = AutoML(df, target_col='target_variable')
ml.run_complete_analysis()
```
