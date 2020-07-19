#!/usr/bin/env python
# coding: utf-8

# In[1]:


https://files.pythonhosted.org/packages/8e/c4/b4ff57e541ac5624ad4b20b89c2bafd4e98f29fd83139f3a81858bdb3815/rake_nltk-1.0.4.tar.gz!pip install rake_nltk


# In[34]:


import pandas as pd
from rake_nltk import Rake
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv('https://query.data.world/s/uikepcpffyo2nhig52xxeevdialfl7')

df = df[['Title', 'Genre', 'Director', 'Actors', 'Plot']]
df.head()


# In[35]:


df['Key_words'] = ""

for index, row in df.iterrows():
    plot = row['Plot']

    r = Rake()
    # extracting words from the text
    r.extract_keywords_from_text(plot)
    # getting the dictionary with keywords as keys and their score as values
    key_words_dict_scores = r.get_word_degrees()
    # assigning the keywords to the column corresponding to the movie
    row['Key_words'] = list(key_words_dict_scores.keys())
df.head()


# In[36]:

# splitting words by commas
df['Genre'] = df['Genre'].map(lambda x: x.split(','))
df['Actors'] = df['Actors'].map(lambda x: x.split(',')[:3]) # only considering the first three actors' names
df['Director'] = df['Director'].map(lambda x: x.split(','))

# creating unique identity names by merging first & last names into a single word, converting to lowercase 
for index, row in df.iterrows():
    row['Genre'] = [x.lower().replace(' ','') for x in row['Genre']]
    row['Actors'] = [x.lower().replace(' ','') for x in row['Actors']]
    row['Director'] = [x.lower().replace(' ','') for x in row['Director']]
df.head()


# In[37]:


# combining 4 lists (4 columns) of key words into 1 sentence under the Characteristics column
df['Characteristics'] = ''
columns = ['Genre', 'Director', 'Actors', 'Key_words']

for index, row in df.iterrows():
    words = ''
    for col in columns:
        words += ' '.join(row[col]) + ' '
    row['Characteristics'] = words
    
# striping white spaces infront and behind, replacing multiple whitespaces (if any)
df['Characteristics'] = df['Characteristics'].str.strip().str.replace('   ', ' ')

df = df[['Title','Characteristics']]
df.head()


# In[39]:


# instantiating and generating the count matrix
count = CountVectorizer()
count_matrix = count.fit_transform(df['Characteristics'])
count_matrix


# In[40]:


# generating the cosine similarity matrix
cosine_sim = cosine_similarity(count_matrix, count_matrix)
cosine_sim


# In[41]:


indices = pd.Series(df['Title'])
indices[:5]


# In[42]:


# creating a function that takes as argument a movie title, and outputs the top 10 recommendations
def recommend(title, cosine_sim = cosine_sim):
    # intializing an empty list of the recommended movies
    recommended_movies = []

    # getting the index of the movie with the matching title
    idx = indices[indices == title].index[0]

    # creating a series with the similarity scores in descending order
    score_series = pd.Series(cosine_sim[idx]).sort_values(ascending = False)

    # getting indices of the 10 most similar movies
    top_10_indices = list(score_series.iloc[1:11].index)

    # populating the list with the title of the top 10 recommendations
    for i in top_10_indices:
        recommended_movies.append(list(df['Title'])[i])
    return recommended_movies


# In[45]:


recommend('Butch Cassidy and the Sundance Kid')

