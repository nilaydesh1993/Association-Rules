"""
Created on Sun Apr 26 12:11:33 2020
@author: DESHMUKH
ASSOCIATION RULES
"""
#conda install -c conda-forge mlxtend
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from mlxtend.frequent_patterns import apriori,association_rules
pd.set_option('display.max_columns',12)

# ===================================================================================
# Business Problem :- Association Rules on Book Dataset
# ===================================================================================

book = pd.read_csv("book.csv")
book.shape
book.head()
book.isnull().sum()
book.info()
book.columns

# Appling Apriori Rules
Book_apr = apriori(book, min_support=0.015, max_len=4, use_colnames=True)  

# Most Frequent item sets based on support (Sorting)
Book_apr.sort_values('support', ascending=False, inplace=True)

# Graphical Representation  
plt.bar(x = list(range(0,11)),height = Book_apr.support[0:11],color='rgmyk');plt.xticks(list(range(0,11)),Book_apr.itemsets[0:11],rotation=90)
plt.xlabel('item-sets');plt.ylabel('support')
plt.subplots_adjust(bottom=0.3, top=0.99) # Custom the subplot layout

# Obtaining Association rules
rules = association_rules(Book_apr, metric="lift", min_threshold=1)
rules.head(20)
rules.sort_values('lift',ascending = False, inplace = True)
rules.head(10)

# Creating a csv file 
#rules.to_csv("book_rules.csv",encoding="utf-8")

               #--------------------------------------------#