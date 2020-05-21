
### Visualising clustering

This can be done with:
- t-SNE which creates a 2D map
- Hierarchical clustering, visualized for example by dendrograms. There are two types:
  - agglomeration hierarchical clustering
  - divisive hierarchical clustering
  
To compute hierarchical clustering we will use the library *SciPy* from python. And we will have to import from `scipy.cluster.hierarchy` 
two packages called `dendrogram` and `linkage`.

Example of dendrogram and linkage:
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
![graph1](https://github.com/afclopes/Machine-Learning/blob/master/linkage_graph1.svg)
