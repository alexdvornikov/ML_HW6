# HW6 Pokemon PCA
Unsupervised transfer learning with PCA on Pokemon.

The goal of this homework is to understand principal component analysis, one of the most widely used methods for exploratory data analysis, data visualization, and dimensionality reduction. 

# Instructions:
1. Run PCA on the MNIST handwritten digit data set. You should NOT use sklearn or other packages that automate the process. Use numpy methods (you can use the np.linalg.eig function to compute the eigenvectors). An example of this analysis is provided: https://github.com/peterjsadowski/pokemon_pca
    1. Download the MNIST data set. One option is using the following code.
    ```
    import tensorflow_datasets as tfds
    # Use Tensorflow Datasets to download data.
    mnist_data = tfds.load("mnist")
    # Convert sample of tf dataset to numpy.
    sample = tfds.as_numpy(mnist_data["train"].batch(60000)).__next__() # Get sample.
    images = sample['image'] # images.shape = 60000,28,28,1)
    labels = sample['label'] # labels.shape = (60000, )
    ```
    1. Plot the sorted eigenvalues of the covariance matrix.
    1. Visualize the top 10 principal components (as images). Describe them in words. Do the principal components involve many pixels or just a few? Do any seem to correspond to particular classes?
    1. Plot reconstructions of some images using k principal components. Approximately how many principal components do you need in order to recognize the digits?

1. Transfer learning: use the representation you learned from MNIST to reconstruct Pokemon. 
    1. Take images of pokemon, embed them in the MNIST PCA space, then map back into image space. The pokemon data set has been preprocessed as grayscale images of size 28x28.
https://github.com/peterjsadowski/pokemon_pca/blob/master/data/pokemon_mnist/pokemon_mnist_images.csv
    1. Do this for different values values of k. Visualize the Pokemon reconstructions, and compare them with the reconstructions generated when the Pokemon principal components are used for the compression: https://github.com/peterjsadowski/pokemon_pca/blob/master/PokemonPCA.ipynb. 
    1. Plot the Pokemon reconstruction error as a function of k for both methods (Pokemon PCA and MNIST PCA). 


# To turn in:
A jupyter notebook containing your analysis, *submitted via github.*





