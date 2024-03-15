#!/usr/bin/env python
# coding: utf-8

# # Data Visualization and Predictive Analytics Assignment 2
# Submitted By: Amruth Karun M V
# <br>RollNo: 2020MCS120004
# <br>Date: 18-Apr-2022

# ### JSE Financial Data Visualization and Regression Analysis
# ### Introduction
# The objective of this assignment is to analyse a dataset that contains correlated variables, visualize the given dataset using different charts, describe the relevance of these visualizations, what insights we get by using the particular graph and perform simple linear regression for any two correlated variables, determine all parameters including confidence intervals, and Model valuation by checking all assumptions, constructing the relevant plots and also by testing of hypothesis.
# 
# ### Dataset
# The dataset used for the assignment is the Johannesburg Stock Exchange dataset which contains details about 50 non-financial firms with numerous financial metrics which are often used by the financial experts to value these companies. This provides us information about which companies are the biggest players in the market, which company has the biggest chance of being backrupt, the past data allows us to make total revenue, future stock predictions etc. and perform other financial analysis. It consist of 400 rows and 21 columns and contains details of the 50 firms from 2010 to 2017.

# In[91]:


# import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns


# In[92]:


data = pd.read_csv('JSE Dataset Aug 2019.csv')
data.head(n=8)


# In[93]:


print("Dataset Shape: ", data.shape)

# Get a summary of the numeric attributes in the dataset 
# (min, max, mean, std etc)
data.describe()


# ### Data Visualization
# Here, we use different types of graphs and charts to understand the relationship between the data points, their distribution, composition and derive insights from the data to identify the financial status, stock market trends and which companies are the major contributors in the Johannesburg market.
# 
# ### 1. Horizontal Bar Plot of Net Profit in Each Year

# In[94]:


plt.figure(figsize = ( 10 , 8 ))

yearwise_net_profit = data.groupby(by='year').agg({'netprofit':sum}).reset_index()
yearwise_net_profit = yearwise_net_profit.sort_values('netprofit')

# create bar plot for the net profit in each year
plt.barh(yearwise_net_profit['year'].apply(str), yearwise_net_profit['netprofit'],
        color ='orange', height=0.4)
 
plt.xlabel("Year")
plt.ylabel("Net Profit")
plt.title("Net profit by the firms in each year")
plt.show()


# The above horizontal bar plot gives an idea about the total net profit from all firms from 2010 to 2017. Here, the graph is sorted in the increasing order of net profit. This gives us a clear picture about the year with the maximum and the year with the minimum profit. The maximum profit is recorded for the year 2016 and the minimum profit is in the following year 2017.

# ### 2. Histogram for BVPS Distribution

# In[95]:


# Creating histogram
fig, axs = plt.subplots(1, 1, figsize =(6, 6), tight_layout = True)
 
# Remove axes splines
for s in ['top', 'bottom', 'left', 'right']:
    axs.spines[s].set_visible(False)

# Remove x, y ticks
axs.xaxis.set_ticks_position('none')
axs.yaxis.set_ticks_position('none')
   
# Add padding between axes and labels
axs.xaxis.set_tick_params(pad = 5)
axs.yaxis.set_tick_params(pad = 10)
 
# Add x, y gridlines
axs.grid(visible = True, color ='grey',
        linestyle ='-.', linewidth = 0.5,
        alpha = 0.6)

# Creating histogram
N, bins, patches = axs.hist(data['BVPS'], bins=10)
 
# Setting color
fracs = ((N**(1 / 5)) / N.max())
norm = colors.Normalize(fracs.min(), fracs.max())
 
for thisfrac, thispatch in zip(fracs, patches):
    color = plt.cm.viridis(norm(thisfrac))
    thispatch.set_facecolor(color)

# Adding extra features   
plt.xlabel("BVPS")
plt.ylabel("Value")
plt.title('Histogram for BVPS Distribution')
 
# Show plot
plt.show()


# Book value per share (BVPS) takes the ratio of a firm's common equity divided by its number of shares outstanding. This indicates the firm's net asset value on a per share basis. When a firm gets liquidated, BVPS is the amount that the shareholders get and this values is used to calculate the share price of the firm. Here, in above distribution we can see that majority of the BVPS values lie closer to zero and some are in the range 50 to 100. When the BVPS value increases, the stock is perceived as more valuable and the stock price should increase. We can have a better understanding of the company value by comparing BVPS against the share price.

# ### 3. Box Plot for Share Price in Each Year

# In[96]:


sns.set_theme(style="ticks")


f, ax = plt.subplots(figsize=(10, 9))

# create boxplot
sns.boxplot(x = 'sharepr', 
            y = 'year', 
            data = data, 
            hue = data['div'], orient='h', 
            palette = 'vlag')


ax.xaxis.grid(True)
ax.set(ylabel="Year", xlabel="Price", title="Box Plot for Share Price in Each Year")
sns.despine()


# Boxplot gives us information about the data distribution, mean, median and the outliers in the data across different groups. Here we have the yearwise distribution of the share price and highlights the stocks with no dividends. From this distribution we can see that the share price range is near hundred in most of the cases and we can observe a maximum share price in the year 2014. These high values can be observed as outliers in black which is far away from the actual distribution of data.

# ### 4. Density Plot for Showing the Distribution of Debts

# In[97]:


# Create density plot for debt data
density_plot = sns.displot(x=data["tdebt"], hue=data["div"], 
            kind="kde", fill=True)

density_plot.set(xlabel="Total Debt", 
                 title="Density Plot for Showing the Distribution of Debts")
sns.despine()


# The above density plot shows the distribution of total debt of these firms when there is dividend and when there is no dividend. This is a smoothed version of the histogram using kernel density estimations. From the above density plot we can see that the debt distribution is high in a particular region and the density decreases as the debt increases and in these high debt areas are with dividend payouts. This might be because of **dividend recapitalization** where the company takes on new debt in order to pay a special dividend to private investors or shareholders.

# ### 5. Pie Chart for Year-wise Retained Earnings

# In[98]:


yearwise_earnings = data.groupby(by='year').agg({'retearnings':sum}).reset_index()

f, ax = plt.subplots(figsize=(7, 6))

# Year wise contribution to retained earnings
plt.pie(yearwise_earnings["retearnings"], labels = yearwise_earnings["year"])
ax.set(title="Pie Chart for Year-wise Retained Earnings")
# show plot
plt.show()


# Pie chart is used to represent the compositon of data. Here, the composition of retained earnings from each year is visualized. Retained earnings or earnings surplus is the amount that a company has left after paying all the direct and indirect costs, taxes and dividends to the shareholders. From the above pie chart we see the proportion of the earnings in each year and we can observe higher values from 2014 - 2017. This increase in the retained earnings indicates increased profits for the last 4 years in the dataset.

# ### 6. Line Plot of Stock Price Variations from 2010 to 2017

# In[99]:


f, ax = plt.subplots(figsize=(10, 8))

# Line plot for first 5 firms 
plt.plot( 'year', 'sharepr', data=data[(data.Company == "AFRICAN MEDIA")], 
         color='olive', label="AFRICAN MEDIA" )
plt.plot( 'year', 'sharepr', data=data[(data.Company == "ARCELORMITTAL")], 
         color='blue', linewidth=2, label="ARCELORMITTAL")
plt.plot( 'year', 'sharepr', data=data[(data.Company == "ARGENT")], 
         color='orange', linewidth=2, label="ARGENT")
plt.plot( 'year', 'sharepr', data=data[(data.Company == "ASPEN")], 
         color='green', linewidth=2, label="ASPEN")
plt.plot( 'year', 'sharepr', data=data[(data.Company == "BIDVEST")], 
         color='red', linewidth=2, label="BIDVEST")

ax.set(title="Line Plot of Stock Price Variations from 2010 to 2017",
      xlabel="Year", ylabel="Share Price")

plt.legend()
plt.show()


# The above line plot is used to visualize the trends in share prices of the first five non-financial firms in the JSE dataset. There is a big increase in the stock prices of ASPEN and BIDVEST and for other firms, the rise in stock prices are either low or it is gradually decreasing from 2010 to 2017 (blue and yellow lines). For ASPEN the price becomes steady after a while but the prices of BIDVEST dropped in 2016. This type of line plots are really useful in representing the general trend in data and we can use it to compare the financial trends of different companies we are interested in.

# ### 7. Violin Plot for EPS in Each Year

# In[100]:


plt.figure(figsize = ( 10 , 8 ))

# Create violin plot for EPS distribution
violin_plot = sns.violinplot(x=data["year"], y=data["EPS"])

violin_plot.set(title="Violin Plot for EPS in Each Year")
sns.despine()


# The above violin plot shows the distribution of EPS in each year. Earnings Per Share (EPS) is the companies net profit by number of common shares it has outstanding. This value can be used to estimate how much money the company makes for each share of its stock. From this violin plot we can observe the density of data at each point. For most of the companies, the EPS value is close to zero and a small proportion goes to the negative side also. In 2014 and 2015 we can see very high EPS values which indicates high profits.

# ### 8. Scatter Plot Between Operating Income and Market Cap

# In[101]:


plt.figure(figsize = ( 10 , 8 ))

plt.scatter(data["opincome"], data["mktcap"], color = '#88c999')
plt.title("Scatter Plot Between Operating Income and Market Cap")
plt.ylabel("Market Cap")
plt.xlabel("Operating Income")
plt.show()


# The above scatter plot shows the relationship between Market Capitalization and Operating Income. Market cap is the most recent market value of the company's outstanding shares. This value indicates how much the company is valued and how much potential it has. While the operating income indicates how much of the company's revenue will eventually become profit and is a key indicator of the performance of the company and has a positive relationship with the market cap as we can see in the scatter plot.

# In[ ]:




