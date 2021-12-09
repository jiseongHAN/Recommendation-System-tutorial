import numpy as np
import pandas as pd
from ast import literal_eval
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD



####content based filtering

data = pd.read_csv("movies_metadata.csv")

## 필요한 데이터만 남기기
data = data[['id', 'genres', 'vote_average', 'vote_count', 'popularity', 'title', 'overview']]
'''
vote average -> 불공평한 부분이 있음
Weighted Rating 사용
WR = (v / (v+m)) * R + (m / (v+m)) * C
where
     v : number of votes
     R = average Rating
     m = minimum votes to be listed in top 250 (or something)
     C = average rating for whole movies
'''

# m 찾기

m = data["vote_count"].quantile(0.99)
truncated_data = data.loc[data["vote_count"] >= m]

C = truncated_data["vote_average"].mean()

def get_WR(x, m=m, C=C):
    assert 'vote_count' and 'vote_average' in x.keys()
    v = x['vote_count']
    r = x['vote_average']

    wr = (v / (v+m)) * r + (m / (v+m)) * C
    return wr

#data 전처리
truncated_data["score"] = truncated_data.apply(get_WR, axis= 1)
truncated_data['genres'] = truncated_data['genres'].apply(literal_eval)
truncated_data['genres'] = truncated_data['genres'].apply(lambda x: [d['name'] for d in x]).apply(lambda x: " ".join(x))

##  vecotrization
count_vector = CountVectorizer(ngram_range=(1,3))
c_vector_genres = count_vector.fit_transform(truncated_data['genres'])
genre_c_sim = cosine_similarity(c_vector_genres, c_vector_genres).argsort()[:,::-1]

def get_recommend_moive_list(df, movie_title, top=30):
    df.index = range(df.shape[0])
    target_moive_index = df[df['title'] == movie_title].index.values

    sim_index = genre_c_sim[target_moive_index, : top].reshape(-1)
    sim_index = sim_index[sim_index != target_moive_index]

    result = df.iloc[sim_index].sort_values("score", ascending=False)[:10]

    return result

#### collaborative filtering

#item based
rating_data = pd.read_csv("small_movies/ratings.csv")
movie_data = pd.read_csv("small_movies/movies.csv")

rating_data.drop("timestamp", axis = 1, inplace=True)

user_movie_rating = pd.merge(rating_data, movie_data, on='movieId')

movie_user_rating = user_movie_rating.pivot_table("rating", index="title", columns="userId")
user_movie_rating = user_movie_rating.pivot_table("rating", index = "userId", columns="title")

movie_user_rating.fillna(0, inplace=True)
user_movie_rating.fillna(0, inplace=True)

item_based_collaborative = cosine_similarity(movie_user_rating)
item_based_collaborative = pd.DataFrame(data= item_based_collaborative, index = movie_user_rating.index, columns =movie_user_rating.index)

def get_item_based_collabor(title):
    return item_based_collaborative[title].sort_values(ascending=False)[:6]

get_item_based_collabor("'Hellboy': The Seeds of Creation (2004)")

#Latent factor based 
rating_data = pd.read_csv("small_movies/ratings.csv")
movie_data = pd.read_csv("small_movies/movies.csv")

rating_data.drop("timestamp", axis = 1, inplace=True)
movie_data.drop("genres", axis=1, inplace=True)

user_movie_data = pd.merge(rating_data, movie_data, on='movieId')

user_movie_rating = user_movie_data.pivot_table("rating", index="userId", columns="title").fillna(0)

SVD = TruncatedSVD(12)
matrix = SVD.fit_transform(movie_user_rating)

corr = np.corrcoef(matrix)


movie_title = user_movie_rating.columns
movie_title_list = list(movie_title)

coffey_hands = movie_title_list.index("'Hellboy': The Seeds of Creation (2004)")
corr_coffey_hands = corr[coffey_hands]
list(movie_title[(corr_coffey_hands >= 0.9)])[:50]