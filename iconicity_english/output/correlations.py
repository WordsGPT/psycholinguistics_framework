import pandas as pd
#import seaborn as sns
#import matplotlib.pyplot as plt
import numpy as np
#from matplotlib.colors import ListedColormap
#import matplotlib.colors as mcolors

method = 'pearson'  # method correlation

print(f"Using method: {method}")

# Read data
df = pd.read_excel('iconicity_ratings_english_joined.xlsx')

# Select iconicity columns
iconicity_cols = ['rating', 'iconicity_english_base_prompt_gpt-4o', 'iconicity_english_base_prompt_gpt-4o_finetuned', 'iconicity_english_UPM-LLM']
df_iconicity = df[iconicity_cols].apply(pd.to_numeric, errors='coerce')

# Correlations
corr = df_iconicity.corr(method=method)

print(corr)
corr.to_excel("correlations_upm-llm.xlsx", index=False)