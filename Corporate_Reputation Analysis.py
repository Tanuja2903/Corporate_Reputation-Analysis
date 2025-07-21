#!/usr/bin/env python
# coding: utf-8

# ## **Loading Data**

# In[ ]:


from google.colab import files
import io
import pandas as pd


def upload():
  uploaded = files.upload()
  filename = next(iter(uploaded))
  df = pd.read_csv(io.BytesIO(uploaded[filename]),lineterminator='\n')
  return df


# ### Before Layoff Data

# In[ ]:


tweets_df = upload()
tweets_df.head()


# ### Layoff Data

# In[ ]:


layoffs_df = upload()
layoffs_df.head()


# ### After Layoff Data

# In[ ]:


after_tweets_df = upload()
after_tweets_df.head()


# ## **Loading Trained ML Model**

# In[ ]:


import pickle

uploaded = files.upload()
filename = next(iter(uploaded))
model = pickle.load(open(filename, 'rb'))


# ## **Data Preprocessing**

# ### Importing Libraries

# In[ ]:


## Data Manipulation
import pandas as pd
import numpy as np
import re

## Text Preprocessing
get_ipython().system('pip install contractions')
import contractions

import nltk
nltk.download('stopwords') #for stopwords
nltk.download('wordnet') #for WordNetLemmatizer
nltk.download('punkt') #for word_tokenize

from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from sklearn.feature_extraction.text import TfidfVectorizer

## Data Visualization
import seaborn as sns
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

import warnings
warnings.filterwarnings("ignore")


# ### Deleting unwanted columns

# In[ ]:


def del_col(df):
  print(df.columns)
  return df.drop(['User', 'Date Created'], axis=1)


# In[ ]:


tweets_df = del_col(tweets_df)
layoffs_df = del_col(layoffs_df)


# In[ ]:


layoffs_df = layoffs_df.rename(columns={'Tweet\r': 'Tweet'})


# In[ ]:


after_tweets_df = del_col(after_tweets_df)


# In[ ]:


after_tweets_df = after_tweets_df.rename(columns={'Tweet\r': 'Tweet'})


# ### Data cleaning
# 
# 1.   Lower casing
# 2.   Removal of Urls
# 3.   Removal of @tags and #
# 4.   Removal of punctuations
# 5.   Removal of emojis and symbols
# 6.   Removal of stop words
# 7.   Lemmatization
# 
# 
# 
# 
# 

# In[ ]:


def data_cleaning(tweet):
  # covert all text to lowercase
  tweet = tweet.lower()

  # remove all urls
  tweet = re.sub('http\S+|www\S+|https\S+', '', tweet, flags=re.MULTILINE)

  # remove @ user tags and #
  tweet = re.sub('\@\w+|\#', '', tweet)

  # remove emojis
  regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
                           "]+", flags = re.UNICODE)
  regrex_pattern.sub('',tweet)

  # remove numbers
  tweet = ''.join(c for c in tweet if not c.isdigit())

  # resolving contractions
  expanded = []
  for word in tweet.split():
    expanded.append(contractions.fix(word))
  tweet =  ' '.join(expanded)

  # remove punctuations
  tweet = re.sub('[^\w\s]', '', tweet)

  # remove stop words
  tweet_tokens = word_tokenize(tweet)
  filtered_texts = [word for word in tweet_tokens if word not in stop_words]

  # lemmatizing
  lemma = WordNetLemmatizer()
  lemma_texts = (lemma.lemmatize(text, pos='a') for text in filtered_texts)

  return " ".join(lemma_texts)


# In[ ]:


tweets_df.Tweet = tweets_df['Tweet'].apply(data_cleaning)
layoffs_df.Tweet = layoffs_df['Tweet'].apply(data_cleaning)


# ### Checking for duplicate rows and deleting them

# In[ ]:


def drop_dupli(df):
  duplicate = df[df.duplicated()]
  return df.drop_duplicates('Tweet')


# In[ ]:


tweets_df = drop_dupli(tweets_df)
layoffs_df = drop_dupli(layoffs_df)
print(tweets_df.shape,layoffs_df.shape)


# In[ ]:


after_tweets_df = drop_dupli(after_tweets_df)
print(after_tweets_df.shape)


# ## **Predicting sentiments**

# In[ ]:


def pred_senti (df,proc_df):
  vect = TfidfVectorizer(sublinear_tf=True).fit(proc_df['Tweet'].values.astype('U'))
  tweets = vect.transform(df['Tweet'])
  sentiment = model.predict(tweets)
  return sentiment


# ### Loading x_train that was used for vectorising

# In[ ]:


vect_df = upload()
vect_df.head()


# In[ ]:


tweets_df['Sentiment'] = pred_senti(tweets_df, vect_df)
layoffs_df['Sentiment'] = pred_senti(layoffs_df, vect_df)


# In[ ]:


# Drop rows with NaN values in the 'Tweet' column
after_tweets_df.dropna(subset=['Tweet'], inplace=True)

# Now apply the pred_senti function
after_tweets_df['Sentiment'] = pred_senti(after_tweets_df, vect_df)


# ## **Calculating Corporate Reputation**
# 
# 

# In[ ]:


def Calculate_NBR(df):
  pos = df[df.Sentiment == 'Positive']
  pos_count = len(pos.index)
  neg = df[df.Sentiment == 'Negative']
  neg_count = len(neg.index)
  NBR = ((pos_count-neg_count)/(pos_count+neg_count))*100
  return NBR


# In[ ]:


NBR_before_layoff = Calculate_NBR(tweets_df)
NBR_only_layoff = Calculate_NBR(layoffs_df)
NBR_after_layoff = Calculate_NBR(after_tweets_df)
print(NBR_before_layoff,NBR_after_layoff,NBR_only_layoff)


# ## **Data Visualization**

# ### Bar graph of NBR scores

# In[ ]:


Dataset = ['Before Layoff', 'After Layoff', 'After (layoff tweets)']
values = [NBR_before_layoff,NBR_after_layoff,NBR_only_layoff]
fig = plt.figure(figsize = (5, 5))

# creating the bar plot
plt.bar(Dataset, values, color ='maroon', width = 0.4)

plt.xlabel("Datasets")
plt.ylabel("Net Brand Reputation")
plt.title("Net Brand Reputation Comparisons")
plt.show()


# ### Count graphs of various datasets

# In[ ]:


fig, ax = plt.subplots(1,3,figsize=(10,5))
fig.tight_layout(pad=4.0)
ax[0].title.set_text('Before Layoff')
ax[1].title.set_text('After Layoff')
ax[2].title.set_text('After Layoff (layoff tweets)')
sns.countplot(x = 'Sentiment', data = tweets_df, ax=ax[0])
sns.countplot(x = 'Sentiment', data = after_tweets_df, ax=ax[1])
sns.countplot(x = 'Sentiment', data = layoffs_df, ax=ax[2])
fig.show()


# ### Pie chart of various datasets

# In[ ]:


fig, ax = plt.subplots(1,3, figsize=(10,10))
fig.tight_layout(pad=4.0)
colors = ("green","yellow","red")
tags = tweets_df['Sentiment'].value_counts()
tags1 = after_tweets_df['Sentiment'].value_counts()
tags2 = layoffs_df['Sentiment'].value_counts()
explode = (0.1,0.1,0.1)
tags.plot(kind='pie', autopct='%1.1f%%', shadow=True, colors = colors,
          startangle=90, explode = explode, label='', ax=ax[0],
          title='Before Layoff')
tags1.plot(kind='pie', autopct='%1.1f%%', shadow=True, colors = colors,
          startangle=90, explode = explode, label='', ax=ax[1],
           title='After Layoff')
tags2.plot(kind='pie', autopct='%1.1f%%', shadow=True, colors = colors,
          startangle=90, explode = explode, label='', ax=ax[2],
           title='After Layoff (layoff tweet)')

