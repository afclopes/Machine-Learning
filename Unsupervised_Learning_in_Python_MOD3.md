## Dimension Reduction

Dimension reduction finds patterns in data and presents it in a compressed form. It carries out this so call compression by removing 
noise (or less informative features) from the data. This noise is often what causes problems when carrying out *supervised learning* 
such as prediction tasks like classification and regression. Thus, dimension reduction is what makes preditions possible.

### Principle Component Analysis (PCA)

PCA is a type of dimension reduction. It is based on principal components, or directions of variance. These directions are measured 
and presented in a numpy array with each column as a principal component.

This is divided into 2 steps:
- decorrelation
- reduces dimension

What does PCA do?
It rotates the data (more precisely the directions of variance) so that it becomes aligned to the axes. And it brings the means of the 
samples to 0. Very little information here is lost. 

In scikit-learn, PCA has a `.fit()` and a `.transform()` function, so that the data can be run through `.fit()` first, and be transformed, 
and then the transformed data is run through the `.transform()` function. This means that new data can be run through the `.transform()` 
function later too.

PCA is imported from the library `sklearn.decomposition`, and linear correlation of the data samples is computed with Pearson correlation, 
which takes values from 1 to -1, with larger values indicating strong correlations, and 0 indicates no correlation. 

Example:
```python
# Perform the necessary imports
import matplotlib.pyplot as plt
from scipy.stats import pearsonr 

# Assign the 0th column of grains: width
width = grains[:,0]

# Assign the 1st column of grains: length
length = grains[:,1]

# Scatter plot width vs length
plt.scatter(width, length)
plt.axis('equal')
plt.show()

# Calculate the Pearson correlation
correlation, pvalue = pearsonr(width,length)
```
![](https://github.com/afclopes/Machine-Learning/blob/master/images/pca_graph0.svg)

