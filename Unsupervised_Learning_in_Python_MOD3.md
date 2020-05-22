## Dimension Reduction

Dimension reduction finds patterns in data and presents it in a compressed form. It carries out this so call compression by removing noise (or less informative features) from the data. This noise is often what causes problems when carrying out *supervised learning* such as prediction tasks like classification and regression. Thus, dimension reduction is what makes preditions possible.

### Principle Component Analysis (PCA)

PCA is a type of dimension reduction. It is based on principal components, or directions of variance. These directions are measured and presented in a numpy array with each column as a principal component.

This is divided into 2 steps:
- decorrelation
- reduces dimension

What does PCA do?
It rotates the data (more precisely the directions of variance) so that it becomes aligned to the axes. And it brings the means of the samples to 0. Very little information here is lost. 

In scikit-learn, PCA has a `.fit()` and a `.transform()` function, so that the data can be run through `.fit()` first, and be transformed, and then the transformed data is run through the `.transform()` function. This means that new data can be run through the `.transform()` function later too.

PCA is imported from the library `sklearn.decomposition`, and linear correlation of the data samples is computed with Pearson correlation, which takes values from 1 to -1, with larger values indicating strong correlations, and 0 indicates no correlation. 

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

Example of decorrelation: 
```python
# Import PCA
from sklearn.decomposition import PCA

# Create PCA instance: model
model = PCA()

# Apply the fit_transform method of model to grains: pca_features
pca_features = model.fit_transform(grains)

# Assign 0th column of pca_features: xs
xs = pca_features[:,0]

# Assign 1st column of pca_features: ys
ys = pca_features[:,1]

# Scatter plot xs vs ys
plt.scatter(xs, ys)
plt.axis('equal')
plt.show()

# Calculate the Pearson correlation of xs and ys
correlation, pvalue = pearsonr(xs, ys)

# Display the correlation
print(correlation)
```

```python
<script.py> output:
    2.5478751053409354e-17
 ```

![](https://github.com/afclopes/Machine-Learning/blob/master/images/pca_graph1.svg)

**Intrinsic dimension:** number of features required to approximate the dataset. It informs us about how much the data can be compressed. It can be identified by counting the PCA features that have high variance. Usually the more PCAs are computed, the smaller the variance of that PCA. As such, the first PCA will have the highest variance, and the last PCA will have the smallest variance. It can be tricky to decide which variances are considered high, and this will depend on where you set the threshold. Below this threshold is what will be considered as "noise", and above is what will be considered as informative data. But sometimes this does not work well. We will see more of this later.

Example of finding first PCA:
```python
# Make a scatter plot of the untransformed points
plt.scatter(grains[:,0], grains[:,1])

# Create a PCA instance: model
model = PCA()

# Fit model to points
model.fit(grains)

# Get the mean of the grain samples: mean
mean = model.mean_

# Get the first principal component: first_pc
first_pc = model.components_[0,:]

# Plot first_pc as an arrow, starting at mean
plt.arrow(mean[0], mean[1], first_pc[0], first_pc[1], color='red', width=0.01)
```
![](https://github.com/afclopes/Machine-Learning/blob/master/images/pca_graph2.svg)
The graph shows the direction in which the data varies the most.

Example to find the intrinsic dimension:
```python
# Perform the necessary imports
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

#data has too many rows, need to standardize features first
# Create scaler: scaler
scaler = StandardScaler()

# Create a PCA instance: pca
pca = PCA()

# Create pipeline: pipeline
pipeline = make_pipeline(scaler,pca)

# Fit the pipeline to 'samples'
pipeline.fit(samples)

# Plot the explained variances
features = range(pca.n_components_) #use the pca components as a range for the bar plot of variances
plt.bar(features, pca.explained_variance_)
plt.xlabel('PCA feature')
plt.ylabel('variance')
plt.xticks(features)
plt.show()
```
![](https://github.com/afclopes/Machine-Learning/blob/master/images/pca_graph3.svg)
Here PCA features 0 and 1 have the biggest variance, so there is an intrinsic dimension of 2 of the data.

Example of PCA with 2 components:
```python
# Import PCA
from sklearn.decomposition import PCA

# Create a PCA model with 2 components: pca
pca = PCA(n_components=2)

# Fit the PCA instance to the scaled samples
pca.fit(scaled_samples)

# Transform the scaled samples: pca_features
pca_features = pca.transform(scaled_samples)

# Print the shape of pca_features
print(pca_features.shape)
```

When PCA does not manage to explain the data well, then we need to use `csr_matrix` which will store only the non-zero values. But scikit-learn PCA does not support `csr_matrix`, which means that we have to use a replacement called `TruncatedSVD` which runs a PCA and can work with `csr_matrix`. 

One common way in which `csr_matrix` are formed is from word documents, and using `TfidfVectorizer`, where `Tf` stands for the frequency of the word in the document, and `idf` refers to a reduction of the influence of frequent words. 

Let's look at how a csr_matrix is created on word documents, making word frequency arrays:
```python
# Import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Create a TfidfVectorizer: tfidf
tfidf = TfidfVectorizer() 

# Apply fit_transform to document: csr_mat
csr_mat = tfidf.fit_transform(documents)

# Print result of toarray() method
print(csr_mat.toarray())

# Get the words: words
words = tfidf.get_feature_names()

# Print words
print(words)
```

Output:
```python
<script.py> output:
    [[0.51785612 0.         0.         0.68091856 0.51785612 0.        ]
     [0.         0.         0.51785612 0.         0.51785612 0.68091856]
     [0.51785612 0.68091856 0.51785612 0.         0.         0.        ]]
    ['cats', 'chase', 'dogs', 'meow', 'say', 'woof']
```

Now let's run a PCA based on crs_matrix:
```python
# Perform the necessary imports
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline

# Create a TruncatedSVD instance: svd
svd = TruncatedSVD(n_components=50)

# Create a KMeans instance: kmeans
kmeans = KMeans(n_clusters=6)

# Create a pipeline: pipeline
pipeline = make_pipeline(svd,kmeans)

# Import pandas
import pandas as pd

# Fit the pipeline to articles
pipeline.fit(articles)

# Calculate the cluster labels: labels
labels = pipeline.predict(articles)

# Create a DataFrame aligning labels and titles (a list of article titles): df
df = pd.DataFrame({'label': labels, 'article': titles})

# Display df sorted by cluster label
print(df.sort_values('label'))
```

Output: Can you identify any patterns?
```python
<script.py> output:
        label                                        article
    59      0                                    Adam Levine
    57      0                          Red Hot Chili Peppers
    56      0                                       Skrillex
    55      0                                  Black Sabbath
    54      0                                 Arctic Monkeys
    53      0                                   Stevie Nicks
    52      0                                     The Wanted
    51      0                                     Nate Ruess
    50      0                                   Chad Kroeger
    58      0                                         Sepsis
    30      1                  France national football team
    31      1                              Cristiano Ronaldo
    32      1                                   Arsenal F.C.
    33      1                                 Radamel Falcao
    37      1                                       Football
    35      1                Colombia national football team
    36      1              2014 FIFA World Cup qualification
    38      1                                         Neymar
    39      1                                  Franck Ribéry
    34      1                             Zlatan Ibrahimović
    26      2                                     Mila Kunis
    28      2                                  Anne Hathaway
    27      2                                 Dakota Fanning
    25      2                                  Russell Crowe
    29      2                               Jennifer Aniston
    23      2                           Catherine Zeta-Jones
    22      2                              Denzel Washington
    21      2                             Michael Fassbender
    20      2                                 Angelina Jolie
    24      2                                   Jessica Biel
    10      3                                 Global warming
    11      3       Nationally Appropriate Mitigation Action
    13      3                               Connie Hedegaard
    14      3                                 Climate change
    12      3                                   Nigel Lawson
    16      3                                        350.org
    17      3  Greenhouse gas emissions by the United States
    18      3  2010 United Nations Climate Change Conference
    19      3  2007 United Nations Climate Change Conference
    15      3                                 Kyoto Protocol
    8       4                                        Firefox
    1       4                                 Alexa Internet
    2       4                              Internet Explorer
    3       4                                    HTTP cookie
    4       4                                  Google Search
    5       4                                         Tumblr
    6       4                    Hypertext Transfer Protocol
    7       4                                  Social search
    49      4                                       Lymphoma
    42      4                                    Doxycycline
    47      4                                          Fever
    46      4                                     Prednisone
    44      4                                           Gout
    43      4                                       Leukemia
    9       4                                       LinkedIn
    48      4                                     Gabapentin
    0       4                                       HTTP 404
    45      5                                    Hepatitis C
    41      5                                    Hepatitis B
    40      5                                    Tonsillitis
```

