import pandas as pd
#import seaborn as sns
#import matplotlib.pyplot as plt
import numpy as np
#from matplotlib.colors import ListedColormap
#import matplotlib.colors as mcolors

method = 'pearson'  # method correlation

print(f"Using method: {method}")

# Read data
df = pd.read_excel('test_split_joined.xlsx')

# Select iconicity columns
iconicity_cols = ['rating', 'iconicity_new']
df_iconicity = df[iconicity_cols].apply(pd.to_numeric, errors='coerce')
df_iconicity.columns = ['human',  'test_dataset']

# Correlations
corr = df_iconicity.corr(method=method)

print(corr)
corr.to_excel("correlations_test_dataset.xlsx", index=False)