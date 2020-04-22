import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#taking the dataset 

dataset = pd.read_csv("movie_dataset.csv")

#feature selection

features = ['genres','keywords','cast','director']

for i in features:
    dataset[i] = dataset[i].fillna("")

def combine_features(row):
    return row['genres']+" "+row['keywords']+" "+row['cast']+" "+row['director']


dataset["combined_features"] = dataset.apply(combine_features,axis=1)

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()

count_matrix = cv.fit_transform(dataset["combined_features"])
    
from sklearn.metrics.pairwise import cosine_similarity

cosine_sim = cosine_similarity(count_matrix)



def get_title_from_index(index):
    return dataset[dataset.index == index]["title"].values[0]

def get_index_from_title(title):
    return dataset[dataset.title == title]["index"].values[0]



movie_user_likes = input("Enter your favourite movie :")

movie_index = get_index_from_title(movie_user_likes)

similar_movies = list(enumerate(cosine_sim[movie_index]))

sorted_similar_movies = sorted(similar_movies,key = lambda x:x[1],reverse = True)

i=0

for movie in sorted_similar_movies:
    print(get_title_from_index(movie[0]))
    i = i+1
    if i>50:
        break
    