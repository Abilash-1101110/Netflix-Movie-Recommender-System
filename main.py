# import necessary libraries 
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#load the dataset 
movies=pd.read_csv("top10K-TMDB-movies.csv")
movies.head(5)

#to see the description of dataset 
movies.describe()

#to gather the information about  the dataset  
movies.info()

# to check weather there's any null value's present 
movies.isnull()

#feeatures selection 
movies.columns

#id,genere,title,overview
#create another dataset from the derived features
movies=movies[['id','title','genre','overview']]

#for content based recommender system we need text so merging of 2 columns 
movies['tags']=movies['overview']+movies['genre']

#now remove 2 coloumns 
new_data=movies.drop(columns=['overview','genre'])

print(new_data)

# using BOw model 
cv=CountVectorizer(max_features=10000,stop_words='english')
print(cv)
#fit and transform to whole data to convert text into some data
vector=cv.fit_transform(new_data['tags'].values.astype('U')).toarray()
print(vector)
#recommender system building using cosine similarity 
similarity=cosine_similarity(vector)
print(similarity)

#acessing the movie name using their index value 
n=new_data[new_data['title']=='The Godfather'].index[0]
print(n)


#calculate the distance and enumerate it so that we can gain the top recomenndation 
distance=sorted(list(enumerate (similarity[2])),reverse=True ,key=lambda vector:vector[1])
for i in distance[0:5]:
    print(new_data.iloc[i[0]].title)


#creating a function 
def recommand(movies):
    index=new_data[new_data['title']==movies].index[0]
    distance = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda vector:vector[1])
    for i in distance[0:5]:
        print(new_data.iloc[i[0]].title)

recommand("Iron Man")

#import pickle for saving the model 
import pickle
pickle.dump(new_data, open('movies_list.pkl', 'wb'))
pickle.dump(similarity, open('similarity.pkl', 'wb'))
pickle.load(open('movies_list.pkl', 'rb'))