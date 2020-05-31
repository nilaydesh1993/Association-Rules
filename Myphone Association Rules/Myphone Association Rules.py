"""
Created on Sun Apr 26 16:14:33 2020
@author: DESHMUKH
ASSOCIATION RULES
"""
#conda install -c conda-forge mlxtend
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from mlxtend.frequent_patterns import apriori,association_rules

# ===================================================================================
# Business Problem :- Association Rules on Myphonedata Data.
# ===================================================================================

myphonedata = pd.read_csv("myphonedata.csv")
myphonedata = myphonedata.iloc[:,3:]
myphonedata.isnull().sum()
myphonedata.shape
myphonedata.head()

# Summary
myphonedata.describe()

# Appling Apriori Rules
myphonedata_apr = apriori(myphonedata, min_support = 0.001, use_colnames = True)

# Most Frequent item sets based on support (Sorting)
myphonedata_apr.sort_values('support', ascending = False, inplace = True)
myphonedata_apr.head(10)

# Graphical Representation
plt.bar(x = list(range(0,11)),height = myphonedata_apr.support[0:11],color='rgmbk');plt.xticks(list(range(0,11)),myphonedata_apr.itemsets[0:11],rotation=90)
plt.xlabel('item-sets');plt.ylabel('support')
plt.subplots_adjust(bottom = 0.3, top = 0.99) # Custom the subplot layout

# Obtaining Association Rules
Rules = association_rules(myphonedata_apr, metric = "lift", min_threshold = 1)
Rules.sort_values('lift', ascending = False, inplace = True)
Rules.head(10)

# Creating a csv file 
#Rules.to_csv("myphonedata.csv",encoding="utf-8")

                       #--------------------------------------------#





