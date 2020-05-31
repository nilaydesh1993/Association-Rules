"""
Created on Sun Apr 26 20:22:54 2020
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

transactions = pd.read_csv("transactions_retail1.csv",header = None)
df = transactions.apply(lambda s:s.str.replace("'", ""))
df = df.replace(np.nan , 0)

# Removing Special symbols and numbers
df.replace(regex=True, inplace=True, to_replace=r'[0-9,(+*),.]', value=r'0')

# Converting Dataframe into list
transaction_list = []
for i in range (0,len(df)):
    transaction_list.append ([str(df.values[i,j]) for j in range(0,6) if str(df.values[i,j])!='0'])

transaction_list [0]

# Converting List to a Dataframe for the transactions data 
transaction_series  = pd.DataFrame(pd.Series(transaction_list))

# Giving Title as Transactions
transaction_series.columns = ["transactions"] 

# Creating a dummy columns for the each item in each transactions ... Using column names as item name
X = transaction_series['transactions'].str.join(sep='*').str.get_dummies(sep='*')

# Appling Apriori Rules
transaction_apr = apriori(X, min_support=0.015, max_len=4, use_colnames=True)  

# Most Frequent item sets based on support (Sorting)
transaction_apr.sort_values('support', ascending=False, inplace=True)

# Graphical Representation  
plt.bar(x = list(range(0,11)),height = transaction_apr.support[0:11],color='rgmyk');plt.xticks(list(range(0,11)),transaction_apr.itemsets[0:11],rotation=90)
plt.xlabel('item-sets');plt.ylabel('support')
plt.subplots_adjust(bottom=0.3, top=0.99) # Custom the subplot layout

# Obtaining Association rules
rules = association_rules(transaction_apr, metric="lift", min_threshold=1)
rules.head(20)
rules.sort_values('lift',ascending = False, inplace = True)
rules.head(10)

# Creating a csv file 
#rules.to_csv("transaction_rules.csv",encoding="utf-8")

                          #--------------------------------------------#
















