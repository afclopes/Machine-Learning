### Non-negative matrix factorization (NMF)

NMF is also a dimension reduction technique. It is considered to be more interpretable than PCAs, because it decomposes samples as sums of its parts. But all data used must be zero or bigger than zero.

Things to know about NMFs:
- also uses `.fit()` and `.transform()`
- must be given specified number of components to test
- can work with NumPy arrays and `csr_matrix` arrays

Similar to PAC, NMF is also imported from the library `sklearn.decomposition`, and also has components, and the dimensions of these components is the same as the dimensions of the sample. So the array will be formed of columns which refer to the different PCAs, and rows which will refer to the data tested, eg. a word in a document, and all data in the array will be non-negative.

We can reconstruct back the original sample by multiplying components by feature values, and then adding those up. This leads to a product of matices as a result. Here is an example: 
The following array is the components of our model:
```python
[[1.  0.5 0. ]
 [0.2 0.1 2.1]]
```
These are the NMF feature values: `[2,1]`

The first row of the array will be multiplied by the first number of the feature values, to give: `[2. 1. 0.]`
And the same will happen with the second row to give: `[0.2 0.1 2.1]`
So now our array is like:
```python
[[2.  1. 0. ]
 [0.2 0.1 2.1]]
```
Now we must add the columns together to give: `[2.2 1.1 2.1]` And this likely represents the original data.

Example applying NMF:
```python
# Import NMF
from sklearn.decomposition import NMF

# Create an NMF instance: model
model = NMF(n_components=6)

# Fit the model to articles
model.fit(articles)

# Transform the articles: nmf_features
nmf_features = model.transform(articles)

# Print the NMF features
print(nmf_features)
# Import pandas
import pandas as pd

# Create a pandas DataFrame: df
df = pd.DataFrame(nmf_features,index=titles)

# Print the row for 'Anne Hathaway'
print(df.loc['Anne Hathaway']) #here we are requesting to see an NMF feature for an article about Anne Hathaway

# Print the row for 'Denzel Washington'
print(df.loc['Denzel Washington'])
```

Output:
```python
<script.py> output:
    0    0.003845
    1    0.000000
    2    0.000000
    3    0.575711
    4    0.000000
    5    0.000000
    Name: Anne Hathaway, dtype: float64
    0    0.000000
    1    0.005601
    2    0.000000
    3    0.422380
    4    0.000000
    5    0.000000
    Name: Denzel Washington, dtype: float64
```
This out put shows that the NMF feature 3 has the highest value, this means that both of these features were computed using mainly the 3rd NMF component.

To understand bettwe NMF components, let's think in terms of documents. NMF components are topics of documents, and NMF features combines topics into documents. 

Example to see what was the topic of the 3rd NMF component:
```python
# Import pandas
import pandas as pd

# Create a DataFrame: components_df
components_df = pd.DataFrame(model.components_,columns=words)

# Print the shape of the DataFrame
print(components_df.shape)

# Select row 3: component
component = components_df.iloc[3]

# Print result of nlargest
print(component.nlargest())
```

Output:
```python
<script.py> output:
    (6, 13125)
    film       0.627877
    award      0.253131
    starred    0.245284
    role       0.211451
    actress    0.186398
    Name: 3, dtype: float64
```
So the topic that Anne Hathaway and Denzel Washington have most in common is film.

If we think in terms of images, then NMF components will refer to patterns of frequently occuring images. 

Images are composed of cells with different levels of grey, which are measured as brightness ranging from 0 (black) to 1 (white). And greyscale images can be represented as flat arrays. Flat array happens when a 2D array is read row-by-row and then from left-to-right, and top-to-bottom, creating a 1D array. Then, each row will refer to an image, and each column will refer to a pixel.

To reconstruct back the original data of an image, you need to transform a flattened array into a 2D again. For this you need to use `.reshape((,))` and specify the dimensions of this new array.

Example of reconstructing an image:
```python
# Import pyplot
from matplotlib import pyplot as plt

# Select the 0th row: digit
digit = samples[0,:] #since samples here is a NumPy array, you can't use .loc[] or .iloc[]

# Print digit
print(digit)

# Reshape digit to a 13x8 array: bitmap
bitmap = digit.reshape(13,8)

# Print bitmap
print(bitmap)

# Use plt.imshow to display bitmap
plt.imshow(bitmap, cmap='gray', interpolation='nearest')
```
![](https://github.com/afclopes/Machine-Learning/blob/master/images/NMF_graph0.svg)


Another example of NMF learning parts of images:
```python
# Import NMF
from sklearn.decomposition import NMF

# Create an NMF model: model
model = NMF(n_components=7) #7 is the number of cells in an LED display

# Apply fit_transform to samples: features
features = model.fit_transform(samples)

# Call show_as_image on each component
for component in model.components_:
    show_as_image(component)

# Assign the 0th row of features: digit_features
digit_features = features[0,:]

# Print digit_features
```
The output of this will be 7 different images. The last image is seen below:

![](https://github.com/afclopes/Machine-Learning/blob/master/images/NMF_graph1.svg)

To show the comparison between how NMF and PCA do analysis, let's run an example similar to the one above, but instead of using NMF, let's use PCA:
```python
# Import PCA
from sklearn.decomposition import PCA

# Create a PCA instance: model
model = PCA(n_components=7)

# Apply fit_transform to samples: features
features = model.fit_transform(samples)

# Call show_as_image on each component
for component in model.components_:
    show_as_image(component)
```
The output of this will be 7 different images. The last image is seen below:

![](https://github.com/afclopes/Machine-Learning/blob/master/images/NMF_graph2.svg)

Some of the previous images also show parts in red, which represent negative values computed with the PCA. Here is an example:

![](https://github.com/afclopes/Machine-Learning/blob/master/images/NMF_graph3.svg)


NMF can also be used to make suggestions. For example, you can have a document, and run NMF to find NMF features, and based on these you can find other documents that are similar to the original one. By similar, I mean of a similar topic, because documents can have different frequency of usage of the same set of words. Still, regardless of this frequency, there are ways to detect that these documents are about the same topic. We can do that using a term called cosine similarity. 

Example:
```python
# Perform the necessary imports
import pandas as pd
from sklearn.preprocessing import normalize

# Normalize the NMF features: norm_features
norm_features = normalize(nmf_features)

# Create a DataFrame: df
df = pd.DataFrame(norm_features, index=titles)

# Select the row corresponding to 'Cristiano Ronaldo': article
article = df.loc['Cristiano Ronaldo']

# Compute the dot products: similarities
similarities = df.dot(article)

# Display those with the largest cosine similarity
print(similarities.nlargest())
```

Output:
```python
<script.py> output:
    Cristiano Ronaldo                1.000000
    Franck Ribéry                    0.999972
    Radamel Falcao                   0.999942
    Zlatan Ibrahimović               0.999942
    France national football team    0.999923
    dtype: float64
```

Another example, that takes the frequency that certain artists have been listened to by users, and suggests other artists:
```python
# Perform the necessary imports
from sklearn.decomposition import NMF
from sklearn.preprocessing import Normalizer, MaxAbsScaler
from sklearn.pipeline import make_pipeline

# Create a MaxAbsScaler: scaler
scaler = MaxAbsScaler()  #transforms the data so that all users have the same influence on the model

# Create an NMF model: nmf
nmf = NMF(n_components=20)

# Create a Normalizer: normalizer
normalizer = Normalizer()

# Create a pipeline: pipeline
pipeline = make_pipeline(scaler,nmf,normalizer)

# Apply fit_transform to artists: norm_features
norm_features = pipeline.fit_transform(artists)

# Import pandas
import pandas as pd

# Create a DataFrame: df
df = pd.DataFrame(norm_features,index=artist_names)

# Select row of 'Bruce Springsteen': artist
artist = df.loc['Bruce Springsteen']

# Compute cosine similarities: similarities
similarities = df.dot(artist)

# Display those with highest cosine similarity
print(similarities.nlargest())
```
