import statsmodels.regression.linear_model as lm
import statsmodels.api as sm
import numpy as np

nsample = 100
x = np.linspace(0, 10, 100)
print(x)
X = np.column_stack((x, x ** 2))
print(X)
beta = np.array([1, 0.1, 10])
e = np.random.normal(size=nsample)


X = sm.add_constant(X)
y = np.dot(X, beta) + e

model = lm.OLS(y, X)
results = model.fit()
print(results.summary())



nsample = 50
sig = 0.5
x = np.linspace(0, 20, nsample)
X = np.column_stack((x, np.sin(x), (x - 5) ** 2, np.ones(nsample)))
beta = [0.5, 0.5, -0.02, 5.0]

y_true = np.dot(X, beta)
y = y_true + sig * np.random.normal(size=nsample)