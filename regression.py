#!/usr/bin/env python
# coding: utf-8

# # Data Visualization and Predictive Analytics Assignment 2
# Submitted By: Amruth Karun M V
# <br>RollNo: 2020MCS120004
# <br>Date: 16-Apr-2022

# ### JSE Financial Data Visualization and Regression Analysis
# ### Introduction
# The objective of this assignment is to analyse a dataset that contains correlated variables, visualize the given dataset using different charts, describe the relevance of these visualizations, what insights we get by using the particular graph and perform simple linear regression for any two correlated variables, determine all parameters including confidence intervals, and Model valuation by checking all assumptions, constructing the relevant plots and also by testing of hypothesis.
# 
# ### Dataset
# The dataset used for the assignment is the Johannesburg Stock Exchange dataset which contains details about 50 non-financial firms with numerous financial metrics which are often used by the financial experts to value these companies. This provides us information about which companies are the biggest players in the market, which company has the biggest chance of being backrupt, the past data allows us to make total revenue, future stock predictions etc. and perform other financial analysis. It consist of 400 rows and 21 columns and contains details of the 50 firms from 2010 to 2017.

# In[57]:


# import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns

# In[58]:


data = pd.read_csv('JSE Dataset Aug 2019.csv')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

x = data["opincome"].values.reshape(-1, 1)
y = data["mktcap"].values.reshape(-1, 1)

# Calculate mean and standard deviation
x_bar = np.mean(x)
y_bar = np.mean(y)

std_x = np.std(x)
std_y = np.std(y)

print("Descriptive Statistics:")
print("Mean of x = ", x_bar)
print("Mean of y = ", y_bar)

print("\nStandard deviation of x = ", std_x)
print("Standard deviation of y = ", std_y)

r  = np.corrcoef(data["opincome"], data["mktcap"])[1][0]
r2 = r * r
print("\nCorrelataion coefficient (r) = ", r)
print("r squared value (r2) = ",r2)

''' 
Fit the dataset using the LinearRegression model.
Here the full dataset is used inorder to understand
the relationship between the two selected attributes
and also to determine all the parameters and check the
relationship between correlation coefficient and
coefficient of determination.
'''

lr = LinearRegression()
lr.fit(x , y)
y_predict = lr.predict(x)

# print predicted values
# print(y_predict)

b_1 = lr.coef_[0][0]
b_0 = lr.intercept_[0]
print("\nModel slope (b1) = ", b_1)
print("Model intercept (b0) = ", b_0)
print("Regression Equation (y_cap) = {0} + {1} * x".format(b_0, b_1))

MSE = mean_squared_error(y, y_predict)
# Calculate Root Mean Square Error(RMSE)
RMSE = np.sqrt(MSE)

print("\nMSE value = ", MSE)
print("RMSE value = ", RMSE)

# Calculate coefficient of determination
R2 = r2_score(y, y_predict)
print("\nR2 Score (Coefficient of determination) = ", R2)
print("{:2f} percent variation is explained by the model.".format(R2 * 100))

'''
# Get summary using stats model api
import statsmodels.api as sm
x = sm.add_constant(x.ravel())
results = sm.OLS(y,x).fit()
print(results.summary())
'''

residual = y - y_predict
residual_squared = residual * residual
RSS = sum(residual_squared)[0]
print("\nSum of squared residuals (RSS) = ", RSS)

variation = y - y_bar
variation_squared = variation * variation
TSS = sum(variation_squared)[0]
print("Sum of total variation squared (TSS) = ", TSS)
calculated_R2 = (TSS - RSS) / TSS
print("Calculated R2 value =", calculated_R2)


# For the JSE dataset, we perform Simple Linear Regression with *opincome* as independant variable(x) and *mktcap* as dependend variable (y). We use LinearRegression from *sklearn* to fit our dataset. The mean of operating income is equal to **2462132121.275** and mean of market cap is **29375451863.09**. Their standard deviations are **7208530172.24** and **59912103482.32** respectively. The correlation coefficient (r) is calculated to be **0.917**. This indicates a high positive correlation between operating income and market cap. After fitting the model with our data, the obtained slope **b1 = 7.62** and model intercept **b0 = 10608904757.29**. Therefore, the regression equation is **y_cap = 10608904757.291008 + 7.62207151421611 * x**. This means for every unit increase in operating income there is a 7.62 times increase in market cap and when the operating income is zero, the market cap is a constant value equal to the intercept. <br>
# 
# The MSE and RMSE values are also calculated from the predicted y value which is 5.70e+20 and 23887746371.97 respectively. The residual values are calculated by subtracting the predicted values from the observed market cap. The sum of squared residuals (RSS) and sum of squared to total variation (TSS) is calculated and from that the R2 score is calculated as **(TSS - RSS) / TSS**. The calculated coefficient of determination (R2) value is observed to be **0.84102**. This means 84 percent of the variation is explained by the model. We want the R2 value as high as possible i.e close to 1. Here only 16 percent of the variation is not explained by the model which can be due to other factors or random variation. 

# ### Standard Error Estimates

# In[71]:


N = len(residual)
degrees_of_freedom = N - 2

# Calculate standard error estimates
sigma = np.sqrt(RSS/degrees_of_freedom)
print("Regression standard error(sigma) = ", sigma)

x_bar_square = x_bar**2
sum_x_diff_square = sum((x - x_bar)**2)
sigma_beta0 = sigma * np.sqrt((1/N) + (x_bar_square / sum_x_diff_square))[0]
print("Standard error estimate of Beta0 =", sigma_beta0)

sigma_beta1 = sigma / np.sqrt(sum_x_diff_square)[0]
print("Standard error estimate of Beta1 =", sigma_beta1)


# The standard error estimates are used to calculate the accuracy of predictions made with the regression line. Here, we calculate the standard error estimate sigma as **23947690622.73**, for Beta0 as **1265302901.86** and for Beta1 as **0.1661**. The smaller the value of a standard error of estimate, the closer are the dots to the regression line and better is the estimate based on the equation of the line. Here, the calculated error estimates are higher, this shows the variation in our predictions.

# ### Confidence Intervals and Hypothesis Testing
# We construct a 95% confidence interval for the best estimation of the above parameters. The interval is the set of values for which a hypothesis test to the level of 5% cannot be rejected.

# In[72]:


from scipy import stats

# Significance level
alpha = 0.05
print("Significance level (alpha) =", alpha)
print("Degrees of freedom =", degrees_of_freedom)
# Calculate critical value for two-tailed test
t_value = stats.t.ppf(1 - (alpha/2), degrees_of_freedom)
print("Critical value (t_alpha_by_two, n-2) =",t_value)

print("\nModel intercept (b0) = ", b_0)
print("Model slope (b1) = ", b_1)

# Calculate confidence interval for beta values
conf_beta0_right = b_0 + (t_value * sigma_beta0)
conf_beta0_left = b_0 - (t_value * sigma_beta0)
print("\nConfidence interval for Beta0 = [{}, {}]"
      .format(conf_beta0_left, conf_beta0_right))

conf_beta1_right = b_1 + (t_value * sigma_beta1)
conf_beta1_left = b_1 - (t_value * sigma_beta1)
print("Confidence interval for Beta1 = [{}, {}]"
      .format(conf_beta1_left, conf_beta1_right))

# Perform Hypothesis Test
print("\nNull Hypothesis: Beta1 = 0 \nAlternate Hypothesis: Beta1 != 0")

t_statistic = b_1 / sigma_beta1
print("Test statistic t =", t_statistic)

if(t_statistic > t_value):
    print("We Reject the Null Hypotheis."          " The model is statistically significant."          " Beta1 != 0")
else:
    print("The model is not statistically significant."          "We accept the Null Hypothesis. Beta1 = 0")


# The confidence interval for beta0 = [8121392229.871651, 13096417284.710365] and for beta1 = [7.295515493558951, 7.9486275348732685] by conducting a two tailed student t-test with a significance level alpha = 0.05 (95% confidence interval). The critical value (t α/2 ) = **1.965** comes from the student t-distribution with (n – 2) degrees of freedom. We have a sample distribution of 400. i.e the degrees of freedom = 398. Then we perform the hypothesis test compute the test statistic t = **45.88** which is greater than the critical value. This indicates that our model is statistically significant and we can reject the null hypothesis and accept the alternate hypotheis Beta1 != 0. The slope is significantly different from zero and there is a statistically significant relationship between operating income and market cap.

# ### Residual Plot 
# The resiudal value is how much the regression line vertically misses the observed data point. A residual plot has the residual values on the vertical axis and the independent variable on the horizontal axis. If the points in a residual plot are randomly dispersed around the horizontal axis, a linear regression model is appropriate for the data. The residual plot for the model created is given below.

# In[73]:


plt.figure(figsize = ( 10 , 8 ))

# Create the residual plot 
plt.scatter(x = x, y = residual)
plt.title("Residual Plot")
plt.ylabel("Residual")
plt.xlabel("Operating Income")
plt.show()


# ### Residual vs Fits Plot
# Residuals versus fits plot is a scatter plot of residuals on the y axis and fitted values (estimated responses) on the x axis. The plot is used to detect non-linearity, unequal error variances, and outliers.

# In[74]:


plt.figure(figsize = ( 10 , 8 ))

# Create the residual vs fits plot
plt.scatter(x = y_predict, y = residual)
plt.title("Residual vs Fits Plot")
plt.ylabel("Residual")
plt.xlabel("Fitted Value")
plt.show()


# Both the above plots gives the simialr information. From the above residual plots we can see that the pattern formed is not exactly a scatter pattern.  In some areas the the density is higher and there is a fanning out pattern in some portions. This indicates that a linear model may not be appropriate for the given problem and probably we need a non-linear model or a model with higher order of terms in x must be required to understand the relationship between the variables.

# ### Normal P-P Plot
# The P-P plot is a graphical representation used to determine how well a given data set fits a specific probability distribution that we are testing. The normal P-P plot plots the residuals against the expected values of the residuals as if it is taken from a normal distribution. When the residuals are normally distributed, it creates a straight line. The P-P plot compares the observed cumulative distribution function (CDF) of the standardized residual to the expected CDF of the normal distribution.

# In[75]:


# Plot the histogram of the error terms
plt.figure(figsize = ( 10 , 8 ))
sns.displot(residual, bins = 20)
plt.title('Error Terms')
plt.xlabel('Residuals')
plt.show()


# In[76]:


from scipy.stats import norm, rankdata

plt.figure(figsize = ( 10 , 8 ))

# Rank the data with by placing min rank for a tie
ranks = rankdata(residual, method='min')

cumulative_prob = (ranks - 0.5) / N
percent = norm.ppf(cumulative_prob)

plt.xlabel('Residual')
plt.ylabel('Percent') 
plt.title('Normal P-P Plot') 
sns.regplot(x=residual, y=percent)
plt.show()


# From the previous histogram for the residuals we have seen that the distribution is not exactly normal. The above normal plot clearly indicates a curvature. The curvature in both ends of the normal probability plot indicates non-normality. This means that our assumptions are not fully valid and a linear model may not be perfect fit for this problem.
# 
# ### Conclusion
# We have successfully explored the JSE dataset about the 50 non-financial firms frrom 2010 - 2017. This dataset consist of total of 400 rows and 21 columns which includes different financial metrics used to value the company, net profit, debt, equity and other financial indicators. Using different types of plots, we were able to visualize the different aspects of the dataset, understand the composition, the relationship between the attributes, identify the derived attributes and understand the distribution of different attributes. Then we performed simple linear regression to identify the reletionship between market cap and operating income. We can observe a positive relationship between the two variables. For better understanding of the relationship, we determined all the parameters including standard error estimates, residual plots normal plot, computed confidence intervals and performed the hypothesis test to validate our model.

# In[ ]:




