# Machine learning

Machine learning is a subset of artificial intelligence. It encapsulates the idea of learning from data, finding patterns and computing 
calculations or generating more data with minimal influence from humans. There are different branches of machine learning:
- Unsupervised learning: refers to searching for patterns and extracting features of an unlabeled dataset without a prediction
- Supeprvised learning: involves having a labeled dataset which the algorithm will use to learn to predict the output based on the input data


Let's start learning.


## Unsupervised Learning

What it does:
- Looks for patterns in the dataset
- Clusters together based on a similarity
- Compresses data based on a similarity

Some methods of clustering that we will learn:
- K-means clustering:
  - finds clusters
  - numbers of clusters need to be specified
  - will be implemented here with *sklearn*, also known as *scikit-learn*
  - creates a model
  - remembers the means of your samples, also known as centrioids
  

You just got a dataset, you have cleaned it and it's ready to start analysing. First thing, is to check if there is a pattern between 
the two variables that you are interested in. Best way to start investigating this is to make a scatter plot to observe how many 
clusters are formed. 

Next, create a KMeans model that finds these clusters seen with the scatter plot. Label these clusters, and then predict new 
labels with a new dataset:
``` python
# Import KMeans
from sklearn.cluster import KMeans 

# Create a KMeans instance with 3 clusters: model
model = KMeans(n_clusters=3)

# Fit model to points
model.fit(points)

# Determine the cluster labels of new_points: labels
labels = model.predict(new_points)

# Print cluster labels of new_points
print(labels)
```

Present the data produced in a new scatter plot:
```python
# Import pyplot
import matplotlib.pyplot as plt

# Assign the columns of new_points: xs and ys
xs = new_points[:,0]
ys = new_points[:,1]

# Make a scatter plot of xs and ys, using labels to define the colors
plt.scatter(xs,ys,c=labels,alpha=0.5)

# Assign the cluster centers: centroids
centroids = model.cluster_centers_

# Assign the columns of centroids: centroids_x, centroids_y
centroids_x = centroids[:,0]
centroids_y = centroids[:,1]

# Make a scatter plot of centroids_x and centroids_y
#marker type is dimond
#marker size is 50
plt.scatter(centroids_x,centroids_y,marker='D',s=50)
plt.show()
```

![KMeansGraph](https://github.com/afclopes/Machine-Learning/blob/master/images/KMeans_clustering_graph1.svg)

Scatter plots show centroids of model with KMeans fitting in the center of the scatter plot clusters of the new data. But how do 
we know that this is in fact a model that fits well to our new data? How do we know that 3 clusters is the best? Can we evaluate the quality 
of the clustering?

One way is to compare the clusters to the sample data and see if the different clusters correspond to different traits in the data.
We can do this comparison with *cross-tabulation*. In this case you will get a dataframe that you can look at and see if the clusters 
fit well with the data.

The tighter the clusters, the better the quality of the custering. This clustering quality can be measured by inertia, which is the distance 
of the samples to the centroid of the cluster. The lower the spread out, the better the clustering will be. K-means usually tries to 
minimise the inertia when choosing clusters. Picking the number of clusters involves a trade off between numbers and quality. If 
you make a plot of "number of clusters" produced with k-means by "inertia", the 'elbow' of this inertia plot should show the number 
of the clusters most appropriate and in agreement with this trade-off. This will be where the inertia becomes to decrease more slowly, 
but has the least number of clusters.

```python
ks = range(1, 6) #randomly selecting a few numbers of clusters to try
inertias = []

for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters=k)
    
    # Fit model to samples
    model.fit(samples)
    
    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)
    
# Plot ks vs inertias
plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
```

```python
# Create a KMeans model with 3 clusters: model
model = KMeans(n_clusters=3)

# Use fit_predict to fit model and obtain cluster labels: labels
labels = model.fit_predict(samples) #same as using .fit() and then .predict()

# Create a DataFrame with labels and varieties as columns: df
df = pd.DataFrame({'labels': labels, 'varieties': varieties})

# Create crosstab: ct
#this will count the number of times each grain variety coincides with each cluster label
ct = pd.crosstab(df['labels'],df['varieties']) 

# Display ct
print(ct)
```
The dataframe we get from this makes it very easy to see that the clustering carried out has worked well. However this method does 
not always work. Some times the traits that distinguish the data, and are the basis of the clustering, have a big variance, which means 
that there is a big spread in the values. These traits with big variances can have a big impact on the data clustering, so we 
can transform that data so that each feature of the data have equal chance to affect the data. Hence, we need to transform the variances.
 We can do this with `StandardScaler` which makes every feature have a mean of 0 and a variance of 1. These transformed features are 
 known to be *standardized*. Unlike K-means which uses `.fit()` and `.predict()`, StandardScaler uses `.fit()` and `.transform`. 
 After the data has been transformed, you will still have to use it for clustering with k-means. This can be automated using a 
 *sklearn pipeline* where data from one function gets fed into the other automatically. In python, with the help of the sklearn library, 
 this can be done easily by importing `StandardScaler` from `sklearn.preprocessing` and `KMeans` from `sklearn.cluster`, and 
 the `make_pipeline` from `sklearn.pipeline`.
 
 Example to prepare the data before clustering:
 ```python
 # Perform the necessary imports
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Create scaler: scaler
scaler = StandardScaler()

# Create KMeans instance: kmeans
kmeans = KMeans(n_clusters=4)

# Create pipeline: pipeline
pipeline = make_pipeline(scaler,kmeans)
```

Example of clustering:
```sql
# Import pandas
import pandas as pd

# Fit the pipeline to samples
pipeline.fit(samples)

# Calculate the cluster labels: labels
#the data will pass through fit and transform and fit and then predict
labels = pipeline.predict(samples)

# Create a DataFrame with labels and species as columns: df
df = pd.DataFrame({'labels':labels,'species':species})

# Create crosstab: ct
ct = pd.crosstab(df['labels'],df['species'])

# Display ct
print(ct)
```
In the sublibrary `sklearn.pipeline`, where we can find `StandardScaler` which standardizes all the features by removing the means 
and scaling the values to unit of variance, we can also access other functions, such as `Normalizer`, which will create a relative 
scale for each different feature of the data.

Example of normalizing:
```sql
# Import Normalizer
from sklearn.preprocessing import Normalizer

# Create a normalizer: normalizer
normalizer = Normalizer()

# Create a KMeans model with 10 clusters: kmeans
kmeans = KMeans(n_clusters=10)

# Make a pipeline chaining normalizer and kmeans: pipeline
pipeline = make_pipeline(normalizer,kmeans)

# Fit pipeline to the daily price movements
pipeline.fit(movements)
```
