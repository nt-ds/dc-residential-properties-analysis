#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

import pylab
import numpy as np
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

sns.set_style("whitegrid")


# In[2]:


class LinearModeling:
    def adjusted_r_squared(sefl, r_squared, num_samples, num_regressors):
        return 1 - (((1 - r_squared) * (num_samples - 1)) / (num_samples - num_regressors - 1))
    
    
    def linear_model(self, X, y, model_name):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=486554)
        X_train = X_train.drop("SALEDATE", axis=1)
        lm = LinearRegression().fit(X_train, y_train)
        print("Performance on training set:")
        r2_train = lm.score(X_train, y_train)
        print("     R_squared:                ", r2_train)
        print("     Adj R_squared:            ", self.adjusted_r_squared(r2_train, X_train.shape[0], X_train.shape[1]))
        print("Performance on test set:")
        r2_test = lm.score(X_test.drop("SALEDATE", axis=1), y_test)
        print("     R_squared:                ", r2_test)
        print("     Adj R_squared:            ", self.adjusted_r_squared(r2_test, X_test.drop("SALEDATE", axis=1).shape[0], X_test.drop("SALEDATE", axis=1).shape[1]))
        y_test_pred = lm.predict(X_test.drop("SALEDATE", axis=1))
        print("     Root mean squared error:   %.2f" % np.sqrt(mean_squared_error(y_test, y_test_pred)))
        plt.scatter(y_test, y_test_pred, alpha=0.5)
        plt.title(f"Predicted vs. Actual - {model_name}")
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.show()
        residuals = y_test - y_test_pred
        return residuals, y_test_pred, lm, X_test.SALEDATE
    
    
    def lasso_model(self, X, y, model_name):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=486554)
        X_train = X_train.drop("SALEDATE", axis=1)
        lasso = LassoCV(alphas=np.arange(0.0001, 1, 0.001), random_state=4856, cv=3).fit(X_train, y_train)
        print("Performance on training set:")
        r2_train = lasso.score(X_train, y_train)
        print("     R_squared:                ", r2_train)
        print("     Adj R_squared:            ", self.adjusted_r_squared(r2_train, X_train.shape[0], X_train.shape[1]))
        print("Performance on test set:")
        r2_test = lasso.score(X_test.drop("SALEDATE", axis=1), y_test)
        print("     R_squared:                ", r2_test)
        print("     Adj R_squared:            ", self.adjusted_r_squared(r2_test, X_test.drop("SALEDATE", axis=1).shape[0], X_test.drop("SALEDATE", axis=1).shape[1]))
        y_test_pred = lasso.predict(X_test.drop("SALEDATE", axis=1))
        print("     Root mean squared error:   %.2f" % np.sqrt(mean_squared_error(y_test, y_test_pred)))
        plt.scatter(y_test, y_test_pred, alpha=0.5)
        plt.title(f"Predicted vs. Actual - {model_name}")
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.show()
        residuals = y_test - y_test_pred
        return residuals, y_test_pred, lasso
    
    
    def ridge_model(self, X, y, model_name):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=486554)
        X_train = X_train.drop("SALEDATE", axis=1)
        ridge = RidgeCV(alphas=np.arange(0.0001, 1, 0.001), cv=3).fit(X_train, y_train)
        print("Performance on training set:")
        r2_train = ridge.score(X_train, y_train)
        print("     R_squared:                ", r2_train)
        print("     Adj R_squared:            ", self.adjusted_r_squared(r2_train, X_train.shape[0], X_train.shape[1]))
        print("Performance on test set:")
        r2_test = ridge.score(X_test.drop("SALEDATE", axis=1), y_test)
        print("     R_squared:                ", r2_test)
        print("     Adj R_squared:            ", self.adjusted_r_squared(r2_test, X_test.drop("SALEDATE", axis=1).shape[0], X_test.drop("SALEDATE", axis=1).shape[1]))
        y_test_pred = ridge.predict(X_test.drop("SALEDATE", axis=1))
        print("     Root mean squared error:   %.2f" % np.sqrt(mean_squared_error(y_test, y_test_pred)))
        plt.scatter(y_test, y_test_pred, alpha=0.5)
        plt.title(f"Predicted vs. Actual - {model_name}")
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.show()
        residuals = y_test - y_test_pred
        return residuals, y_test_pred, ridge
        
        
    def test_residuals(self, residuals, y_test_pred, model_name):
        residuals.hist(bins=50)
        plt.title(f"Distribution of Residuals - {model_name}")
        plt.xlabel("Residuals")
        plt.ylabel("Count")
        plt.show()
        plt.scatter(residuals, y_test_pred, alpha=0.5)
        plt.xlabel("Residuals")
        plt.ylabel("Predicted Values")
        plt.title(f"Residuals vs. Predicted Values - {model_name}")
        plt.show()
        stats.probplot(residuals, dist="norm", plot=pylab)
        pylab.show()

