# python_project
this is a movie recommendation system, it will recommend you to watch similar movies which you watch lastly
import numpy as np
import pandas as pd
import warnings



warnings.filterwarnings('ignore')


# ### geting data set

columns_name=["user_id","item_id","rating","timestamp"]
df=pd.read_csv('ml-100k/ml-100k/u.data', sep="\t", names=columns_name)

df.head()

df.shape

df['user_id'].nunique() # to find the unique user from the data

df['item_id'].nunique() # to get the unique movies;


movies_titles = pd.read_csv('ml-100k/ml-100k/u.item',sep="\|",header=None)

movies_titles.shape


movies_titles=movies_titles[[0,1]]


movies_titles.columns=['item_id','titles']


movies_titles.head()


df=pd.merge(df, movies_titles, on = "item_id")


df

df.tail()

import matplotlib.pyplot as plt
import seaborn as sns

df.groupby('titles').mean()['rating'].sort_values(ascending=False).head()


df.groupby('titles').count()['rating'].sort_values(ascending=False)


ratings=pd.DataFrame(df.groupby('titles').mean()['rating'])

ratings

ratings['number of ratings']=pd.DataFrame(df.groupby('titles').count()['rating'])


ratings


ratings.sort_values(by='rating', ascending=False)

plt.figure(figsize=(10,6))
plt.hist(ratings['number of ratings'],bins=70)
plt.show()

plt.hist(ratings['rating'],bins=70)
plt.show()


sns.jointplot(x='rating' ,y='number of ratings' ,data=ratings,alpha=0.5)


# ### Creating Movie Recomendation

df.head()

moviemat = df.pivot_table(index="user_id",columns="titles", values="rating")

moviemat.head()

ratings.sort_values('number of ratings', ascending=False).head()

starwar_user_rating =moviemat['Star Wars (1977)']
starwar_user_rating.head()

similar_to_starwars = moviemat.corrwith(starwar_user_rating)


corr_starwars = pd.DataFrame(similar_to_starwars, columns=["correlations"])


corr_starwars.dropna(inplace=True)


corr_starwars.head()


corr_starwars.sort_values('correlations',ascending=False).head(10)


corr_starwars = corr_starwars.join(ratings['number of ratings'])


corr_starwars.head()


corr_starwars[corr_starwars['number of ratings']>100].sort_values('correlations',ascending=False)


def predict_movies(movie_name):
    movie_user_rating =moviemat[movie_name]
    similar_to_movie = moviemat.corrwith(movie_user_rating)
    corr_movie = pd.DataFrame(similar_to_movie, columns=["correlations"])
    corr_movie.dropna(inplace=True)
    
    corr_movie = corr_movie.join(ratings['number of ratings'])
    predictions = corr_movie[corr_movie['number of ratings']>100].sort_values('correlations',ascending=False)
    
    return predictions

perdictions = predict_movies('Titanic (1997)')

perdictions.head()


