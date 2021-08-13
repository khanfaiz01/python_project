# python_project
this is a movie recommendation system, it will recommend you to watch similar movies which you watch lastly
import numpy as np
import pandas as pd
import warnings


# In[2]:


warnings.filterwarnings('ignore')


# ### geting data set

# In[3]:


columns_name=["user_id","item_id","rating","timestamp"]
df=pd.read_csv('ml-100k/ml-100k/u.data', sep="\t", names=columns_name)


# In[4]:


df.head()


# In[5]:


df.shape


# In[6]:


df['user_id'].nunique() # to find the unique user from the data


# In[7]:


df['item_id'].nunique() # to get the unique movies;


# In[8]:


movies_titles = pd.read_csv('ml-100k/ml-100k/u.item',sep="\|",header=None)


# In[9]:


movies_titles.shape


# In[10]:


movies_titles=movies_titles[[0,1]]


# In[13]:


movies_titles.columns=['item_id','titles']


# In[14]:


movies_titles.head()


# In[15]:


df=pd.merge(df, movies_titles, on = "item_id")


# In[16]:


df


# In[17]:


df.tail()


# ### exploratory data analysis

# In[18]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[26]:


df.groupby('titles').mean()['rating'].sort_values(ascending=False).head()
#  group the data     mean value       sort the value in ascending order;


# In[ ]:





# In[29]:


# counting the movies how many times being watched;

df.groupby('titles').count()['rating'].sort_values(ascending=False)


# In[30]:


ratings=pd.DataFrame(df.groupby('titles').mean()['rating'])


# In[31]:


ratings


# In[33]:


ratings['number of ratings']=pd.DataFrame(df.groupby('titles').count()['rating'])


# In[34]:


ratings


# In[35]:


ratings.sort_values(by='rating', ascending=False)


# In[37]:


plt.figure(figsize=(10,6))
plt.hist(ratings['number of ratings'],bins=70)
plt.show()


# In[38]:


plt.hist(ratings['rating'],bins=70)
plt.show()


# In[40]:


sns.jointplot(x='rating' ,y='number of ratings' ,data=ratings,alpha=0.5)


# In[ ]:





# ### Creating Movie Recomendation

# In[41]:


df.head()


# In[46]:


moviemat = df.pivot_table(index="user_id",columns="titles", values="rating")


# In[48]:


moviemat.head()


# In[52]:


ratings.sort_values('number of ratings', ascending=False).head()


# In[56]:


starwar_user_rating =moviemat['Star Wars (1977)']
starwar_user_rating.head()


# In[58]:


similar_to_starwars = moviemat.corrwith(starwar_user_rating)


# In[64]:


corr_starwars = pd.DataFrame(similar_to_starwars, columns=["correlations"])


# In[66]:


corr_starwars.dropna(inplace=True)


# In[68]:


corr_starwars.head()


# In[71]:


corr_starwars.sort_values('correlations',ascending=False).head(10)


# In[74]:


corr_starwars = corr_starwars.join(ratings['number of ratings'])


# In[76]:


corr_starwars.head()


# In[78]:


corr_starwars[corr_starwars['number of ratings']>100].sort_values('correlations',ascending=False)


# In[79]:


def predict_movies(movie_name):
    movie_user_rating =moviemat[movie_name]
    similar_to_movie = moviemat.corrwith(movie_user_rating)
    corr_movie = pd.DataFrame(similar_to_movie, columns=["correlations"])
    corr_movie.dropna(inplace=True)
    
    corr_movie = corr_movie.join(ratings['number of ratings'])
    predictions = corr_movie[corr_movie['number of ratings']>100].sort_values('correlations',ascending=False)
    
    return predictions
    
    
    
    


# In[81]:


perdictions = predict_movies('Titanic (1997)')


# In[83]:


perdictions.head()


# In[ ]:
