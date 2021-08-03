import matplotlib.pyplot as plt
import numpy as np
    
def plot_decision_boundary(model, X, y):
    X_ = X.T
    y_ = y.reshape(X.shape[0], 1)
    # Set min and max values and give it some padding
    x_min, x_max = X.T[0, :].min() - 1, X.T[0, :].max() + 1
    y_min, y_max = X.T[1, :].min() - 1, X.T[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X_[0, :], X_[1, :], c=y, cmap=plt.cm.Spectral)