import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('insurance_claims.csv')

# Basic exploration
print("Dataset Overview:")
print(df.info())
print("\nBasic Statistics:")
print(df.describe())

# Analysis 1: Claims by category
claims_by_type = df.groupby('claim_type')['claim_amount'].agg(['count', 'mean', 'sum'])
print("\nClaims Analysis by Type:")
print(claims_by_type)

# Visualization
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='claim_type', y='claim_amount')
plt.title('Average Claim Amount by Type')
plt.savefig('claims_by_type.png')

# More analysis tomorrow...