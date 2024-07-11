import seaborn as sns
import pandas as pd

df_iris = sns.load_dataset('iris')
df_iris["species"] = df_iris["species"].astype("category")
df_iris['species_num'] = pd.factorize(df_iris['species'])[0] + 1


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create a new figure
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Extracting columns
x = df_iris['sepal_length']
y = df_iris['sepal_width']
z = df_iris['petal_length']

# Scatter plot
scatter = ax.scatter(x, y, z, c=df_iris['species_num'], cmap='viridis', s=50)

# Setting labels
ax.set_xlabel('Sepal Length')
ax.set_ylabel('Sepal Width')
ax.set_zlabel('Petal Length')

# Adding color bar
plt.colorbar(scatter, label='Petal Width')

plt.title('3D Plot of Iris Dataset')
plt.show()

