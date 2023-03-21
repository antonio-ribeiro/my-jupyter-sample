import pandas as pd
import matplotlib.pyplot as plt

# Load the Iris dataset from the UCI Machine Learning Repository
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
iris_data = pd.read_csv(url, header=None, names=column_names)

# Display the first 5 rows of the dataset
print(iris_data.head())

# Basic statistics of the dataset
print(iris_data.describe())

# Visualize the dataset
iris_data.groupby('species').size().plot(kind='bar', ylabel='count', title='Iris Species Count')
plt.show()

# Scatter plot matrix to visualize relationships between features
pd.plotting.scatter_matrix(iris_data, figsize=(10, 10), diagonal='kde')
plt.show()
