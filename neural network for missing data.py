#!/usr/bin/env python
# coding: utf-8

# In[1]:


from elasticsearch import Elasticsearch
import pandas as pd
from beautifultable import BeautifulTable
import re
import numpy as np
from numpy import array
from numpy import asarray
from numpy import zeros
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer # Used for stemming
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer # Add the keras tokenizer for summaries tokenization
from tensorflow.keras.preprocessing.sequence import pad_sequences # Add padding to help the Keras Sequencing
from tensorflow.keras.losses import SparseCategoricalCrossentropy # Loss function being used
from sklearn.model_selection import train_test_split # Train-test split


# In[2]:


# Connect to Elastic search
es = Elasticsearch([{'host': 'localhost', 'port': 9200}])


# In[ ]:


# Check if the connection was successful
if es.ping():
    print("Connection successful!")
else:
    print("Connection error!")


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


# ### DF with user's ratings for train/test preprocessing

# In[5]:


# Create pandas dataframe 'rat_df' from 'BX-Book-Ratings.csv' file
# Delete all the rows from dataframe which have rating = 0
rat_df = pd.read_csv("BX-Book-Ratings.csv")
rat_df = rat_df.where(rat_df['rating'] != 0).dropna()

# Create pandas dataframe 'book_df' from 'BX-Books.csv' file
book_df = pd.read_csv("BX-Books.csv")


# In[6]:


# Create 'avg_rat' dataframe which contains the average rating for each book
avg_rat = rat_df.groupby(['isbn'])['rating'].mean().reset_index(name='avg_rating')


# In[7]:


# Create 'usr_rat' dataframe which contains all of the user's personal ratings
usr_rat = rat_df.where(rat_df['uid'] == user_id).dropna()


# In[8]:


# Create 'new_df' dataframe by merging 'book_df' and 'usr_rat' dataframes
book_df = book_df[['isbn','summary']]
new_df = pd.merge(book_df, usr_rat, on='isbn', how='right')
new_df = new_df[['summary','uid','rating']]
new_df = new_df.where(new_df['summary'] != 0).dropna()


# In[9]:


# Text-cleaning function
def clean_and_reform_data(text):
    delete_items = ["&#39;", "&quot;"]
    for item in delete_items:
        text = text.replace(item, ' ')
    # remove punctuation marks
    text = re.sub('[^A-Za-z0-9]+', ' ', text)
    # make lowercase
    text = text.lower()
    return text


# In[10]:


new_df['Clean'] = new_df['summary'].apply(clean_and_reform_data)


# In[11]:


# Remove stopwords
stop_words = set(stopwords.words('english'))
new_df['WithoutStop'] = new_df['Clean'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))


# In[12]:


# Use English stemmer
stemmer = SnowballStemmer("english")

# Apply stemming
new_df['Stemmed'] = new_df['WithoutStop'].apply(lambda x: ' '.join([stemmer.stem(word) for word in str(x).split()]))


# In[13]:


# Extract the final processed summaries column from 'new_df' dataframe
summaries = new_df["Stemmed"].copy()

# Tokenize the summary texts
token = Tokenizer()
token.fit_on_texts(summaries)
vocab_size = len(token.word_index) + 1
texts = token.texts_to_sequences(summaries) # Integer encode the summaries


# In[14]:


# Add zero padding to text sequences
texts = pad_sequences(texts, padding='post')


# In[15]:


# Load the whole embedding into memory
embeddings_index = dict()
f = open('glove.6B.100d.txt', encoding="utf8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))


# In[16]:


# Create a weight matrix of one embedding for each unique word in summary texts
embedding_matrix = zeros((vocab_size, 100))
for word, i in token.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


# ### Neural network training

# In[17]:


# Define training and testing data
# Testing data is the 20% of the overall data, and training the 80%
textTrain, textTest, ratingTrain, ratingTest = train_test_split(texts, new_df['rating'], test_size=0.2)


# In[18]:


# Create our neural network model for predicting ratings
input_length = texts.shape[1]

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=texts.shape[1], trainable=False))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.0001)))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.0001)))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(11, activation='softmax'))

model.summary()


# In[19]:


# Compile the model
model.compile(loss=SparseCategoricalCrossentropy(from_logits = True), optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(textTrain, ratingTrain, epochs=100, batch_size=32, validation_split = 0.2,
          callbacks=[tf.keras.callbacks.EarlyStopping(monitor="loss", patience=3)])


# In[20]:


loss, accuracy = model.evaluate(textTest, ratingTest) # Get the loss and accuracy based on the tests


# ### Get the search results from Elasticsearch

# In[ ]:


# Run a query in index 'books' of Elasticsearch
res = es.search(index='books', query = {"match": {"book_title": user_search}}, size=10000)


# In[22]:


# Store the results of the query in 'temp' dictionary
temp = {}
predict = {}
for hit in res['hits']['hits']:
    temp[hit['_id']] = [hit['_source']['book_title'], hit['_score']]
    predict[hit['_id']] = hit['_source']['summary']


# ### Get the results ready for prediction phase

# In[23]:


# Convert dictionary to pandas dataframe
predict_df = pd.DataFrame(predict.items(), columns=['isbn', 'summary'])

# Apply text cleaning
predict_df['summary'] = predict_df['summary'].apply(clean_and_reform_data)

# Remove stopwords
predict_df['summary'] = predict_df['summary'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

# Apply stemming
predict_df['summary'] = predict_df['summary'].apply(lambda x: ' '.join([stemmer.stem(word) for word in str(x).split()]))


# In[24]:


# Tokenize the the summary texts
predict = predict_df["summary"].copy()
predict_texts = token.texts_to_sequences(predict)


# In[25]:


# Add zero padding to text sequences
predict_texts = pad_sequences(predict_texts, maxlen=input_length, padding='post')


# ### Predict personal ratings for the results & proceed with the personalized formula

# In[26]:


# Add a column to 'predict_df' dataframe with the predicted rating from the neural network
predict_df['predicted_rating'] = model.predict_classes(predict_texts)


# In[ ]:


# Create a table for better visualization of our data
table = BeautifulTable(maxwidth=120)
table.column_headers = ["BOOK RESULTS", "PERSONALIZED SCORE", "BM25 SCORE", "AVERAGE RATING", "PERSONAL RATING"]
table.set_style(BeautifulTable.STYLE_RST)


# In[28]:


# Repeat for each dictionary 'temp' item
for i in temp.keys():
    bm25_score = temp.get(i)[1]
    if i in avg_rat['isbn'].values: # Check if there is an average rating for the given book
        average = avg_rat.loc[avg_rat['isbn'] == i, 'avg_rating'].values[0]
    else:
        average = 0 # average rating doesn't exist

    if i in usr_rat['isbn'].values: # Check if there is a personal rating for the given book
        personal = usr_rat.loc[(usr_rat['isbn'] == i), 'rating'].values[0]
    else: # personal rating is predicted by the neural network
        personal = float(predict_df.loc[(predict_df['isbn'] == i, 'predicted_rating')].values[0])
        
# ================================= ~ COMPUTE THE PERSONALIZED RANKING FOR EACH CASE ~ ======================================
    
    # ~ CASE 1 ~
    # There is no average rating: personal rating is either positive (>5), or negative (<=5)
    if (average == 0):
        if personal > 5 :
            total_ranking = bm25_score + bm25_score * (1/20) * (personal)
        else:
            total_ranking = bm25_score - bm25_score/(personal) 
    
    # ~ CASE 2 ~
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


# In[30]:


# Print the results
print(table)


# In[ ]:




