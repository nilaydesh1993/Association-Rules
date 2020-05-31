"""
Created on Sun Apr 26 15:03:19 2020
@author: DESHMUKH
ASSOCIATION RULES
"""
#conda install -c conda-forge mlxtend
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mlxtend.frequent_patterns import apriori,association_rules
pd.set_option('display.max_columns',30)

# ===================================================================================
# Business Problem :- Association Rules on My movies Data.
# ===================================================================================

mymovies = pd.read_csv("my_movies.csv")
mymovies.shape
mymovies.describe()
mymovies = mymovies.iloc[:,5:15] 
mymovies.isnull().sum()

# Appling Apriori Rules
mymovies_apr = apriori(mymovies, min_support=0.001, use_colnames=True)

# Most Frequent item sets based on support (Sorting)
mymovies_apr.sort_values('support', ascending=False, inplace=True)

# Graphical Representation
plt.bar(x = list(range(0,20)),height = mymovies_apr.support[0:20]);plt.xticks(list(range(0,20)),mymovies_apr.itemsets[0:20],rotation=90)
plt.xlabel('item-sets');plt.ylabel('support')
plt.subplots_adjust(bottom=0.4, top=0.99) # Custom the subplot layout

# Obtaining Association Rules
Rules = association_rules(mymovies_apr, metric="lift", min_threshold=2)
Rules.sort_values('lift', ascending=False, inplace=True)
Rules.head(10)

# Creating a csv file 
#Rules.to_csv("my_movies_rules.csv",encoding="utf-8")

                #--------------------------------------------#






