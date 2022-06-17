#!/usr/bin/env python
# coding: utf-8

# In[1]:


from elasticsearch import Elasticsearch, helpers
import pandas as pd
from beautifultable import BeautifulTable


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


# Create a pandas dataframe from the csv file 'BX-Books'
df = pd.read_csv("BX-Books.csv")


# In[5]:


# If an index named "books" already exits, delete it
if es.indices.exists(index='books'):
    es.indices.delete(index='books', ignore=400)


# In[6]:


# Create new index named "books"
es.indices.create(index='books', ignore=400)


# In[7]:


# Function that reads the data from the csv
def generator(df):
    for index, line in df.iterrows():
        yield {
            "_index": "books",
            "_type": "_doc",
            "_id": f"{line['isbn']}",
            "_source": line.to_dict(),
        }


# In[ ]:


# Upload data into elastic search
try:
    res = helpers.bulk(es, generator(df))
    print("Upload done!")
except Exception as e:
    pass


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

# User enters a book lemma to search
user_search = input("Enter a lemma to search: ")


# In[10]:


# Run a query in index 'books' of Elasticsearch
res = es.search(index='books', query = {"match": {"book_title": user_search}}, size=10000)


# In[ ]:


# Create a table for better visualization of our data
table = BeautifulTable(maxwidth=120)
table.column_headers = ["BOOK RESULTS", "BM25 SCORE"]
table.set_style(BeautifulTable.STYLE_RST)


# In[12]:


# Store the results of the query in 'temp' dictionary
temp = {}
for hit in res['hits']['hits']:
    temp[hit['_id']] = [hit['_source']['book_title'], hit['_score']]


# In[13]:


# Enter the data in the table
for i in temp.items():
    table.rows.append([i[1][0], i[1][1]])


# In[14]:


# Print the results
print(table)


# In[ ]:




