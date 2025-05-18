# Spherical Teacher-Student Perceptron

Implementation from scratch of a Spherical Teacher-Student Perceptron, using Python and Numpy (and a bit of Scikit-Learn), and Matplotlib for visualization. This project was carried out as part of a research internship at the LISN, under the supervision of François LANDES. It focuses on studying class imbalance in classification tasks, based on a spherical Teacher-Student perceptron architecture. This approach helps analyze how imbalanced training data affects the performance in classification models.

The internship subject was based on a paper co-authored by my advisor: https://arxiv.org/abs/2501.11638

# What does it do ?

The program generates synthetic `Gaussian-distributed data` and assigns class labels using a `Teacher-Student perceptron model`. To make classification more challenging, it is possible to add `noise` to prevent the dataset from being `trivially linearly separable`. The generated dataset is then stored in the `/data` folder.

You can then use any dataset to train your `student`, choosing all `hyper-paramaters` (eta, maxiter, test_size, n_splits, loss and method). The program will train the student using a `cross-validation` by testing 42 different training proportions (ptrain values). You will get a graph with all results of cross-validation on all ptrain values, on the train and test set. The graph will be saved in the `/plots` folder and the results will be saved in the `/results` folder.

The implemented perceptron follows a `spherical normalization` scheme. This means that each weight vector is `scaled by 1/sqrt(D)`, + ensuring weight magnitudes remain stable as the input dimension D increases. This normalization prevents overfitting due to large weight values in high-dimensional spaces.

# How to use it ?

## main.py

You can adjust the following hyperparameters when calling the `exec()` function:
- N: Size of the dataset
- D: Dimension of each data point
- bias: Bias to introduce in the teacher (`0` gives 2 `perfectly balanced` classes, `-1` gives about `85-15 imbalance` (85% -1, 15% +1))
- test_size: Size of the test set when splitting in train/test set
- eta: Learning rate
- maxiter: Maximum number of iterations during the learning of the student. Be careful, a high number will make the program longer very fast
- n_splits: Number of folds in the cross-validation
- noise_std: Noise to apply (`1.0` introduces a basic noise level, preventing the dataset from being trivially linearly separable. `0.0` keeps the data clean without any noise.)
- loss: The loss to use (`hinge`, `perceptron` or `error-counting`. `error-counting` isn't usable on the gradient method)
- method: The method to use to train the student (`gradient` or `langevin`)

If the parameter `parallel` at the beginning of the file is set to `True`, the exec() functions will do simultaneously as much as possible `ptrain` values, depending on the number of threads available on your device. If set to `False`, the computations will be done sequentially (which can be slower for large datasets).

For the `langevin` method, the parameters `T` and `maxiter` can be changed directly in the sub-function `fmethod` in the `langevin-if branch`, in the function `student`, situated in `utils.py` (`T=0.005` and `maxiter=15000` seem to be fit for the kind of data treated here). The parameters passed in the function `apprentissage` will not impact the hyper-parameters of this method.

Here is an example of every function in the main:  
`save_data(N=1000, D=125, bias=-1.0, noise_std=0.0)`: Save a dataset of size 1000x125, with a bias of -1 and no noise  
`X, Y, w, b = fetch_data(N=1000, D=125, bias=-1.0, noise_std=0.0)`: Fetch the dataset (creates it if it doesn't exist)  
`delete_data(N=1000, D=125, bias=-1.0, all=False)`: Delete the dataset (if all=True, delete all datasets)  
`exec(X=X, Y=Y, loss='perceptron', method='gradient', test_size=0.2, eta=0.1, maxiter=150, n_splits=10, bias=-1.0, noise_std=0.0)`: Launch the program for the student, given your parameters

## parameters.py

This module provides functions to train and evaluate models using both gradient descent and Langevin dynamics, with support for three different loss functions. It allows flexible hyperparameter tuning and performance analysis.

# Warnings

For `noise_std`, using a value `closer to 0` will lead to less noise, and a more linearly separable dataset. A value `further from 0` will lead to more noise (which could lead to unexploitable results).

To avoid under- or over-training, keep `N 8 times bigger` than `D` minimum.

The loss `error-counting` cannot be used with the method `gradient`.

To introduce noise, I use a method of perturbating the frontier. But I also implemented one randomly flipping the class of some points. This can be used instead by uncommenting it in the function `teacher` in the file `utils.py` (don't forget to comment the other method if you do so, and change the value of `noise_std`, a good value for this method would be `0.1 or 0.05`).

The program can quickly become long. For instance, using `maxiter=150` and `10 folds` for the cross-validation, the `gradient method` takes approximately `10 minutes` on my machine for a dataset of size `5000x500`. For the same dataset, the `langevin` method will take approximately 6 hours if you keep `maxiter=15000`.

This project uses a `set random_state` to reproduct the results. This parameter can be changed at the top of `utils.py`: `np.random.seed(424242)`

To use this project, make sure you have the required dependencies installed: `pip install numpy matplotlib scikit-learn`

# Some Results

The `graphs` folder contains a selection of plots illustrating the results of the experiments and hyperparameter optimization for various configurations.
