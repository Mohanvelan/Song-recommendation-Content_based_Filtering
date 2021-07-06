# Song-recommendation Using Content_based_Filtering
  
### Dataset 

**Dataset Name**: Spotify Dataset’ 1921–2020	

- This dataset contains over 170,000 songs collected from Spotify Web API and it has 17 attributes.

- It contains the attributes such as song name, artist, duration, tempo, popularity, year, etc. These characteristics of the songs will be used for content-based filtering to recommend the user-prefer songs.
  
  
## Implementation Process

We don’t need to clean the noisy data or null values since it didn’t have null values.

### Min-max normalization 

The dataset has been normalized to have values ranges from 0 to 1 in order to make the process of analysis with ease.
```
def normalize(ind):
 max_d = data2.loc[:,ind].max()
 min_d = data2.loc[:,ind].min()
 new_min = 0
 new_max = 1
 data2.loc[:,ind] = ((data2.loc[:,ind] - min_d) / (max_d - min_d))*(new_max - new_min) + new_min; 

for col in num_data.columns:
  normalize(col) 
```

### Silhouette method for Optimal K value 

Optimal k value has been found by finding Silhouette co-efficient for each value of K ranges from 2 to 10. 

```
for i in range(2,10):
  km = KMeans(n_clusters=i)
  label = km.fit_predict(num_data)
  sil_coeff = silhouette_score(num_data, label, metric='euclidean')
  print('For k= %i, Silhouette score= %.5f'%(i, sil_coeff))
```

### K-means clustering

With predicted value of k=2, we clustered our data by predicting the results.

```
k = 2
km = KMeans(n_clusters=k)
pred = km.fit_predict(num_data)
```

### Song recommendation using Content-based filtering

In this context, it does the process of recommendation by making use of the characteristics of a song such as acousticness, energy, popularity, duration, etc and making similarities in these characteristics.

By finding the similarity in features of songs using Euclidean distance as a similarity measure, the songs have been recommended.


### Reference

[https://www.kaggle.com/artempozdniakov/spotify-data-eda-and-music-recommendation](https://www.kaggle.com/artempozdniakov/spotify-data-eda-and-music-recommendation)


