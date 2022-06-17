#!/usr/bin/env python
# coding: utf-8

# In[1]:


from elasticsearch import Elasticsearch
import pandas as pd
from beautifultable import BeautifulTable
import math


# In[2]:


# Connect to Elastic search
es = Elasticsearch([{'host': 'localhost', 'port': 9200}])


# In[ ]:


# Check if the connection was successful
if es.ping():
    print("Connection successful!")
else:
    print("Connection error!")


# In[4]:


# Create pandas dataframe 'rat_df' from 'BX-Book-Ratings.csv' file
# Delete all the rows from dataframe which have rating = 0
rat_df = pd.read_csv("BX-Book-Ratings.csv")
rat_df = rat_df.where(rat_df['rating'] != 0).dropna()


# In[ ]:


print("""
######################################################################################
          ___           _     ___                  _      ___           _          
  ___ ___| _ ) ___  ___| |__ / __| ___ __ _ _ _ __| |_   | __|_ _  __ _(_)_ _  ___ ©
 / -_)___| _ \/ _ \/ _ \ / / \__ \/ -_) _` | '_/ _| ' \  | _|| ' \/ _` | | ' \/ -_)
 \___|   |___/\___/\___/_\_\ |___/\___\__,_|_| \__|_||_| |___|_||_\__, |_|_||_\___|
                                                                  |___/            

                          __...--~~~~~-._   _.-~~~~~--...__
                        //               `V'               \\ 
                       //                 |                 \\ 
                      //__...--~~~~~~-._  |  _.-~~~~~~--...__\\ 
                     //__.....----~~~~._\ | /_.~~~~----.....__\\
                    ====================\\|//====================
                                        `---`
######################################################################################
""")

# User enters his ID and a book lemma to search
user_id = int(input("Enter your ID: "))
user_search = input("Enter a lemma to search: ")


# In[6]:


# Create 'avg_rat' dataframe which contains the average rating for each book
avg_rat = rat_df.groupby(['isbn'])['rating'].mean().reset_index(name='avg_rating')


# In[7]:


# Create 'usr_rat' dataframe which contains all of the user's personal ratings
usr_rat = rat_df.where(rat_df['uid'] == user_id).dropna()


# In[ ]:


# Run a query in index 'books' of Elasticsearch
res = es.search(index='books', query = {"match": {"book_title": user_search}}, size=10000)


# In[9]:


# Store the results of the query in 'temp' dictionary
temp = {}
for hit in res['hits']['hits']:
    temp[hit['_id']] = [hit['_source']['book_title'], hit['_score']]


# In[ ]:


# Create a table for better visualization of our data
table = BeautifulTable(maxwidth=120)
table.column_headers = ["BOOK RESULTS", "PERSONALIZED SCORE", "BM25 SCORE", "AVERAGE RATING", "PERSONAL RATING"]
table.set_style(BeautifulTable.STYLE_RST)


# In[11]:


# Repeat for each dictionary 'temp' item
for i in temp.keys():
    bm25_score = temp.get(i)[1]
    if i in avg_rat['isbn'].values: # Check if there is an average rating for the given book
        average = avg_rat.loc[avg_rat['isbn'] == i, 'avg_rating'].values[0]
    else:
        average = 0 # average rating doesn't exist

    if i in usr_rat['isbn'].values: # Check if there is a personal rating for the given book
        personal = usr_rat.loc[(usr_rat['isbn'] == i), 'rating'].values[0]
    else:
        personal = 0 # personal rating doesn't exist
        
# ================================= ~ COMPUTE THE PERSONALIZED SCORE FOR EACH CASE ~ ========================================
    
    # ~ CASE 1 ~
    # Both average and personal ratings don't exist
    if (average == 0) & (personal == 0):
        total_ranking = bm25_score - math.log(bm25_score) # Decrease BM25 score logarithmically
    
    # ~ CASE 2 ~
    # There is no personal rating: average rating is either positive (>5), or negative (<=5)
    elif (personal == 0):
        if average > 5 :
            total_ranking = bm25_score + bm25_score * (1/30) * (average)
        else:
            total_ranking = bm25_score - bm25_score/(average) 
    
    # ~ CASE 3 ~
    # There is no average rating: personal rating is either positive (>5), or negative (<=5)
    elif (average == 0):
        if personal > 5 :
            total_ranking = bm25_score + bm25_score * (1/20) * (personal)
        else:
            total_ranking = bm25_score - bm25_score/(personal) 
    
    # ~ CASE 4 ~
    # There are both average and personal ratings
    else:
        # Both personal and average ratings are positive (>5)
        if (personal > 5) & (average > 5):
            total_ranking = bm25_score + bm25_score * ((1/20) * personal + (1/30) * average)
        
        # Personal rating is positive, but average is negative    
        elif (personal > 5) & (average <= 5):
            total_ranking = bm25_score + bm25_score * (1/20) * (personal) - bm25_score/(average) 
        
        # Average rating is positive, but personal is negative
        elif (personal <= 5) & (average > 5):
            total_ranking = bm25_score + bm25_score * (1/30) * (average) - bm25_score/(personal) 
        
        # Both personal and average ratings are negative (<=5)
        else:            
            total_ranking = bm25_score  - bm25_score/(0.6 * personal + 0.4 * average)
    
    # Add a new row of data in the table
    table.rows.append([temp.get(i)[0], total_ranking, bm25_score, average, personal])


# In[ ]:


# Sort the table by the column 'PERSONALIZED SCORE' in descending order
table.sort('PERSONALIZED SCORE', reverse = True)


# In[13]:


# Print the results
print(table)


# In[ ]:




