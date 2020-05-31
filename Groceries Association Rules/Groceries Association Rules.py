"""
Created on Mon Apr 27 15:44:20 2020
@author: DESHMUKH
"""
#conda install -c conda-forge mlxtend
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from mlxtend.frequent_patterns import apriori,association_rules
pd.set_option('display.max_columns',12)

# ===================================================================================
# Business Problem :- Association Rules on Groceries Dataset
# ===================================================================================

groceries = []  #Empty List Groceries
with open("groceries.csv") as f:
    groceries = f.read()
    
# Splitting the data into separate transactions using separator as "\n"
groceries = groceries.split("\n")  #Next line

# Converted Single Entities into a List
groceries_list = []
for i in groceries:
    groceries_list.append(i.split(","))   
    
# Creating Data Frame for the transactions data 
groceries_series  = pd.DataFrame(pd.Series(groceries_list)) 

# Removing the last empty transaction
groceries_series = groceries_series.iloc[:9835,:] 

# Giving name to column
groceries_series.columns = ["transactions"]

# Creating a dummy columns for the each item in each transactions ... Using column names as item name
X = groceries_series['transactions'].str.join(sep='*').str.get_dummies(sep='*')

# Appling Apriori Rules
groceries_apr = apriori(X, min_support=0.015, max_len=4, use_colnames=True)  

# Most Frequent item sets based on support (Sorting)
groceries_apr.sort_values('support', ascending=False, inplace=True)

# Graphical Representation  
plt.bar(x = list(range(0,20)),height = groceries_apr.support[0:20],color='rgmyk');plt.xticks(list(range(0,20)),groceries_apr.itemsets[0:20],rotation=90)
plt.xlabel('item-sets');plt.ylabel('support')
plt.subplots_adjust(bottom=0.4, top=0.99) # Custom the subplot layout

# Obtaining Association rules
rules = association_rules(groceries_apr, metric="lift", min_threshold=1)
rules.head(20)
rules.sort_values('lift',ascending = False, inplace = True)
rules.head(10)

# Creating a csv file 
#rules.to_csv("groceries_rules.csv",encoding="utf-8")

               #--------------------------------------------#
