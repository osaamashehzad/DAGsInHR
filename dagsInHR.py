## Step 1: Generate a hypothetical dataset ##

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set seed for reproducibility
np.random.seed(42)

# Generate data
n = 5000

data = {
    "Employee ID": range(1, n + 1),
    "Age": np.random.randint(18, 65, size=n),
    "Gender": np.random.choice(["Male", "Female", "Non-binary"], size=n),
    "Tenure": np.random.randint(0, 40, size=n),
    "Department": np.random.choice(["Sales", "HR", "Tech", "Finance", "Operations"], size=n),
    "Job Role": np.random.choice(["Manager", "Team Lead", "Staff", "Clerk"], size=n),
    "Salary": np.random.randint(30000, 200000, size=n),
    "Performance Rating": np.random.choice([1, 2, 3, 4, 5], size=n),
    "Education Level": np.random.choice(["High School", "Bachelor's", "Master's", "PhD"], size=n),
    "Marital Status": np.random.choice(["Single", "Married", "Divorced", "Widowed"], size=n),
    "Number of Dependents": np.random.randint(0, 5, size=n),
    "Distance from Home": np.random.randint(1, 50, size=n),
    "Work-Life Balance": np.random.choice([1, 2, 3, 4, 5], size=n),
    "Job Satisfaction": np.random.choice([1, 2, 3, 4, 5], size=n),
    "Promotions": np.random.randint(0, 5, size=n),
    "Manager Quality": np.random.choice([1, 2, 3, 4, 5], size=n),
    "Absenteeism": np.random.randint(0, 20, size=n),  # Absent days per year
    "Engagement": np.random.choice([1, 2, 3, 4, 5], size=n),
    "Work Culture": np.random.choice([1, 2, 3, 4, 5], size=n),
    "Team Culture": np.random.choice([1, 2, 3, 4, 5], size=n),
    "Attrition": np.random.choice(["Yes", "No"], p=[0.15, 0.85], size=n)
}

# Create DataFrame
df = pd.DataFrame(data)


## Step 2: Data Exploration To Identify Potential Correlations and Patterns ##

# Display basic statistics for numerical and categorical data
print(df.describe())
print(df.describe(include=['object', 'category']))

# Encode categorical variables using one-hot encoding
encoded_df = pd.get_dummies(df, drop_first=True)  # Converts categorical variables into dummy/indicator variables

# Calculate the correlation matrix for the entire encoded DataFrame
corr_matrix_all = encoded_df.corr()

# Visualizing the extended correlation matrix
# plt.figure(figsize=(15, 12))
# sns.heatmap(corr_matrix_all, annot=False, cmap='coolwarm')  # Annot set to False for better visibility in large matrices
# plt.title('Extended Correlation Matrix Including Categorical Data')
# plt.show()

# Pair plot for selected variables
# sns.pairplot(df[['Age', 'Tenure', 'Salary', 'Job Satisfaction', 'Performance Rating', 'Attrition']], hue='Attrition')
# plt.show()

import statsmodels.api as sm

# Logistic regression for Attrition including multiple predictors
X = df[['Age', 'Tenure', 'Salary', 'Performance Rating', 'Job Satisfaction']]
X = sm.add_constant(X)  # adding a constant
y = df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)

model = sm.Logit(y, X)
result = model.fit()
# print(result.summary())

## Step 3 DAG Visualization Based on Statistical Analysis

import networkx as nx

# Create a DAG
dag = nx.DiGraph()

# Add nodes and edges based on the analysis
dag.add_edges_from([
    ('Performance Rating', 'Job Satisfaction'),
    ('Job Satisfaction', 'Attrition'),
    ('Work-Life Balance', 'Attrition'),
    ('Distance from Home', 'Attrition'),
    ('Salary', 'Job Satisfaction')
])

# Draw the DAG
pos = nx.spring_layout(dag)
nx.draw(dag, pos, with_labels=True, node_size=5000, node_color='skyblue', font_size=10, font_weight='bold')
plt.title('Directed Acyclic Graph (DAG) for Attrition Analysis')
plt.show()








