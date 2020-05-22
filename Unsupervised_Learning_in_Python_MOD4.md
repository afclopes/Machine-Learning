### Non-negative matrix factorization (NMF)

NMF is also a dimension reduction technique. It is considered to be more interpretable than PCAs, because it decomposes samples as sums of its parts. But all data used must be zero or bigger than zero.

Things to know about NMFs:
- also uses `.fit()` and `.transform()`
- must be given specified number of components to test
- can work with NumPy arrays and `csr_matrix` arrays

Similar to PAC, NMF is also imported from the library `sklearn.decomposition`, and also has components, and the dimensions of these components is the same as the dimensions of the sample. So the array will be formed of columns which refer to the different PCAs, and rows which will refer to the data tested, eg. a word in a document, and all data in the array will be non-negative.

We can reconstruct back the original sample by multiplying components by feature values, and then adding those up. This leads to a product of matices as a result.
