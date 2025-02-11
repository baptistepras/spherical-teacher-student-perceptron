# Spherical Teacher-Student Perceptron

Implementation from scratch of a Spherical Teacher-Student Perceptron, using Python and Numpy (and a bit of Scikit-Learn), and using MatPlotLib for graphics. This project was carried out as part of a research internship on class imbalance in image classification at the LISN, supervised by François LANDES. It focuses on studying class imbalance in image classification using a spherical Teacher-Student perceptron. This approach helps analyze how imbalanced training data affects the performance in classification models.

# What does it do ?

The program generates synthetic `Gaussian-distributed data` and assigns class labels using a `Teacher-Student perceptron model`. To make classification more challenging, `noise` is added to prevent the dataset from being `trivially linearly separable`. The generated dataset is then stored in the `/data` folder.

You can then use any dataset to train your `student`, chosing all `hyper-paramaters` (eta, maxiter, test_size, n_splits, loss and method). The program will train the student using a `cross-validation` and trying `42 different ptrain values`. You will get a graph with all results of cross-validation on all ptrain values, on the train and test set. The graph will be saved in the `/plots` folder.

The implemented perceptron follows a `spherical normalization` scheme. This means that each weight vector is `scaled by 1/sqrt(D)`, ensuring that weight magnitudes remain controlled as the feature dimension D increases. This normalization prevents overfitting due to large weight values in high-dimensional spaces.

# How to use it ?

In main.py, you can adjust the following hyperparameters when calling the `exec()` function:
- N: Size of the dataset
- D: Dimension of each data point
- bias: Bias to introduce in the teacher (`0` gives 2 `perfectly balanced` classes, `-1` gives a `75-25 imbalance` (75% -1, 25% +1))
- test_size: Size of the test set when splitting in train/test set
- eta: Learning rate
- maxiter: Maximum number of iterations during the learning of the student. Be careful, a high number will make the program longer very fast
- n_splits: Number of folds in the cross-validation
- noise_std: Noise to apply (`1.0` introduces a basic noise level, preventing the dataset from being trivially linearly separable)
- loss: The loss to use (`hinge`, `perceptron` or `square`. `square` isn't usable on the gradient method)
- method: The method to use to train the student (`gradient` or `langevin`)

Here is an example of every function in the main:  
`save_data(N=5000, D=500, bias=-1.0, noise_std=1.0)`: Save a dataset of size 5000x500, with a bias of -1 and a noise of 1  
`X, Y, w, b = fetch_data(N=5000, D=500, bias=-1.0, noise_std=1.0)`: Fetch the dataset (creates it if it doesn't exist)  
`delete_data(N=5000, D=500, bias=-1, all=False)`: Delete the dataset (if all=True, delete all datasets)  
`exec(X=X, Y=Y, loss='perceptron', method='gradient', test_size=0.2, eta=0.1, maxiter=100, n_splits=10, bias=-1.0, noise_std=1.0)`: Launch the program for the student, given your parameters

# Warnings

For `noise_std`, using a value `closer to 0` will lead to less noise, and a more linearly separable dataset. A value `further from 0` will lead to more noise (which could lead to unexploitable results).

To avoid under- or over-training, keep `N` about `10 times bigger` than `D` minimum. Replace 10 by the number of folds you put in your cross-validation.

The loss `square` cannot be used with the method `gradient`.

To introduce noise, I use a method of perturbating the frontier. But I also implemented one randomly flipping the class of some points. This can be used instead by uncommenting it in the function `teacher` in the file `utils.py` (don't forget to comment the other method if you do so, and change the value of `noise_std`, a good value for this method would be `0.1 or 0.05`).

The program can quickly become long. For instance, using `maxiter=100` and `10 folds` for the cross-validation, the `gradient method` takes approximately `10 minutes` on my machine for a dataset of size `5000x500`.

This project uses a `set random_state` to reproduct the results. This parameter can be changed at the top of `utils.py`: `np.random.seed(424242)`

To use this project, make sure you have the required dependencies installed: `pip install numpy matplotlib scikit-learn`
