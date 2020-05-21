
### Visualising clustering

This can be done with:
- t-SNE which creates a 2D map
- Hierarchical clustering, visualized for example by dendrograms. There are two types:
  - agglomeration hierarchical clustering
  - divisive hierarchical clustering
  
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

``sql
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

These dendrogams are quite different. this goes to show that different linkage methods give different results.

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
