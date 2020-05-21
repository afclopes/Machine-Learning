
### Visualising clustering

This can be done with:
- t-SNE which creates a 2D map
- Hierarchical clustering, visualized for example by dendrograms. There are two types:
  - agglomeration hierarchical clustering
  - divisive hierarchical clustering
  
#### t-SNE clustering

t-SNE stands for t-distributed stochastic neighbor embedding. It creates a 2D or sometimes 3D map, where the map tries to maintain the nearness of the samples.

To create t-SNE we will import `TSNE` from `sklearn.manifold`, and we will use the `fit_transform()` function to fit the model and transform the data simultaneously. This means that you cannot use new samples with the same model created for your first t-SNE plot. This type of plot also has a different feature called *learning rate* which must be picked for each dataset via trial and error (usually values between 50 and 200 will work). When you end up picking a bad learning rate, it will be obvious because the data points will appear all bunched together. 

Note: t-SNE plot have axis that do not carry any meaning. If you plot 3 t-SNE graphs on the same day, with the same code and the same data, all 3 graphs will look different (but the relative distance between the points and clusters will be maintained).
  
Example:
```python
# Import TSNE
from sklearn.manifold import TSNE

# Create a TSNE instance: model
model = TSNE(learning_rate=200)

# Apply fit_transform to samples: tsne_features
tsne_features = model.fit_transform(samples)

# Select the 0th feature: xs
xs = tsne_features[:,0]

# Select the 1st feature: ys
ys = tsne_features[:,1]

# Scatter plot, coloring by variety_numbers
plt.scatter(xs,ys,c=variety_numbers)
```
![graph0](https://github.com/afclopes/Machine-Learning/blob/master/images/tsne_graph0.svg)

```python
# Import TSNE
from sklearn.manifold import TSNE

# Create a TSNE instance: model
model = TSNE(learning_rate=50)

# Apply fit_transform to normalized_movements: tsne_features
tsne_features = model.fit_transform(normalized_movements)

# Select the 0th feature: xs
xs = tsne_features[:,0]

# Select the 1th feature: ys
ys = tsne_features[:,1]

# Scatter plot
plt.scatter(xs,ys,alpha=0.5)

# Annotate the points
for x, y, company in zip(xs, ys, companies):
    plt.annotate(company, (x, y), fontsize=5, alpha=0.75)
plt.show()
```
![graph0](https://github.com/afclopes/Machine-Learning/blob/master/images/tsne_graph1.svg)

  
#### Hierarchical clustering  
  
To compute hierarchical clustering we will use the library *SciPy* from python. And we will have to import from `scipy.cluster.hierarchy` packages called `dendrogram`, `linkage`, and `fcluster`.

'Complete' linkage means that the distance between the clusters is measured as the maximum distance between the samples (clusters) based on the furthest points of those clusters.

'Single' linkage means that the distance between the clusters is measures as the closest distance between the clusters based on the closest points of those clusters.

Example of dendrogram and linkage with method 'complete':
```python
# Perform the necessary imports
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

# Calculate the linkage: mergings
mergings = linkage(samples, method='complete')

# Plot the dendrogram, using varieties as labels
#note: no need to do plt.dendrogram
dendrogram(mergings,
           labels=varieties,
           leaf_rotation=90,
           leaf_font_size=6,
)
plt.show()
```

![graph0](https://github.com/afclopes/Machine-Learning/blob/master/images/linkage_graph0.svg)

Example of dendrogram and linkage with method 'single':

```python
# Perform the necessary imports
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram

# Calculate the linkage: mergings
mergings = linkage(samples, method='single')

# Plot the dendrogram
dendrogram(mergings,labels=country_names,leaf_rotation=90,leaf_font_size=6)
plt.show()
```
![graph0](https://github.com/afclopes/Machine-Learning/blob/master/images/linkage_graph2.svg)

These dendrogams are quite different. This goes to show that different linkage methods give different results.

Example normalising data differently:
```python
# Import normalize
#note: not Normalizer(), but normalize()
from sklearn.preprocessing import normalize

# Normalize the movements: normalized_movements
normalized_movements = normalize(movements)

# Calculate the linkage: mergings
mergings = linkage(normalized_movements,method='complete')

# Plot the dendrogram
dendrogram(mergings,labels=companies,leaf_rotation=90,leaf_font_size=6)
plt.show()
```
![graph1](https://github.com/afclopes/Machine-Learning/blob/master/images/linkage_graph1.svg)


Apart from visualisations, hierarchical clustering can also be used to find cluster labels at intermediate stages.
This can be done by finding different heights on the dendrogram, which shows the distance between merging clusters

```python
# Perform the necessary imports
import pandas as pd
from scipy.cluster.hierarchy import fcluster

# Use fcluster to extract labels: labels
labels = fcluster(mergings,6,criterion='distance')

# Create a DataFrame with labels and varieties as columns: df
df = pd.DataFrame({'labels': labels, 'varieties': varieties})

# Create crosstab: ct
ct = pd.crosstab(df['labels'],df['varieties'])

# Display ct
print(ct)
```
