from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import numpy as np

X = np.arange(0, 1*np.pi, 0.01)
X = X.reshape(X.shape[0], 1)
X_poly = PolynomialFeatures(degree=2, include_bias=False).fit_transform(X)
Y = np.sin(X).flatten()
noise = np.random.normal(0, 0.1, len(Y))
print Y.shape, noise.shape
Y = Y+noise

clf1 = LinearRegression()
clf1.fit(X, Y)
pred1 = clf1.predict(X)

clf2 = LinearRegression()
clf2.fit(X_poly, Y)
pred2 = clf2.predict(X_poly)

fig, ax = plt.subplots(1)
ax.scatter(X, Y, alpha=0.3)
ax.plot(X, pred1, label="Linear Prediction")
ax.plot(X, pred2, label="Polynomial Prediction")
ax.legend()
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')
textstr = r'$y = sin(x) + \mathit{N}(0, 0.1)$'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
ax.set_title('Goodness of fit of polynomial features to a trigonomic function')
plt.show()