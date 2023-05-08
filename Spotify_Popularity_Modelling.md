```python
# Justin Madsen
# DSC630 / Fadi Alsaleem

# import the boys
import numpy as np
import pandas as pd



import matplotlib as mpl
import matplotlib.pyplot as plt

%matplotlib inline
import seaborn as sns

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

import sklearn
from sklearn.compose import make_column_transformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, accuracy_score
from sklearn.model_selection import cross_val_score, GridSearchCV, RepeatedStratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


import re
import string

pd.options.mode.chained_assignment = None
pd.set_option('display.max_rows', 25)
pd.set_option('display.max_columns', 25)
pd.set_option('display.width', 1000)
```


```python
# https://www.kaggle.com/datasets/prasertk/oil-and-gas-stock-prices
# import the dataset
spotify_df = pd.read_csv('songs_normalize.csv')
```


```python
# did it work?
spotify_df.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>artist</th>
      <th>song</th>
      <th>duration_ms</th>
      <th>explicit</th>
      <th>year</th>
      <th>popularity</th>
      <th>danceability</th>
      <th>energy</th>
      <th>key</th>
      <th>loudness</th>
      <th>mode</th>
      <th>speechiness</th>
      <th>acousticness</th>
      <th>instrumentalness</th>
      <th>liveness</th>
      <th>valence</th>
      <th>tempo</th>
      <th>genre</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Britney Spears</td>
      <td>Oops!...I Did It Again</td>
      <td>211160</td>
      <td>False</td>
      <td>2000</td>
      <td>77</td>
      <td>0.751</td>
      <td>0.834</td>
      <td>1</td>
      <td>-5.444</td>
      <td>0</td>
      <td>0.0437</td>
      <td>0.3000</td>
      <td>0.000018</td>
      <td>0.355</td>
      <td>0.894</td>
      <td>95.053</td>
      <td>pop</td>
    </tr>
    <tr>
      <th>1</th>
      <td>blink-182</td>
      <td>All The Small Things</td>
      <td>167066</td>
      <td>False</td>
      <td>1999</td>
      <td>79</td>
      <td>0.434</td>
      <td>0.897</td>
      <td>0</td>
      <td>-4.918</td>
      <td>1</td>
      <td>0.0488</td>
      <td>0.0103</td>
      <td>0.000000</td>
      <td>0.612</td>
      <td>0.684</td>
      <td>148.726</td>
      <td>rock, pop</td>
    </tr>
  </tbody>
</table>
</div>




```python
# get a baseline for the shape
spotify_df.shape
```




    (2000, 18)




```python
# let's convert the genres into a list rather than a basic string
def genre_to_list(df):
    df['genre_list'] = df['genre'].apply(lambda x: x.split(" "))
    return df
```


```python
# conversion time
spotify_df = genre_to_list(spotify_df)

# did it work?
spotify_df.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>artist</th>
      <th>song</th>
      <th>duration_ms</th>
      <th>explicit</th>
      <th>year</th>
      <th>popularity</th>
      <th>danceability</th>
      <th>energy</th>
      <th>key</th>
      <th>loudness</th>
      <th>mode</th>
      <th>speechiness</th>
      <th>acousticness</th>
      <th>instrumentalness</th>
      <th>liveness</th>
      <th>valence</th>
      <th>tempo</th>
      <th>genre</th>
      <th>genre_list</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Britney Spears</td>
      <td>Oops!...I Did It Again</td>
      <td>211160</td>
      <td>False</td>
      <td>2000</td>
      <td>77</td>
      <td>0.751</td>
      <td>0.834</td>
      <td>1</td>
      <td>-5.444</td>
      <td>0</td>
      <td>0.0437</td>
      <td>0.3000</td>
      <td>0.000018</td>
      <td>0.355</td>
      <td>0.894</td>
      <td>95.053</td>
      <td>pop</td>
      <td>[pop]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>blink-182</td>
      <td>All The Small Things</td>
      <td>167066</td>
      <td>False</td>
      <td>1999</td>
      <td>79</td>
      <td>0.434</td>
      <td>0.897</td>
      <td>0</td>
      <td>-4.918</td>
      <td>1</td>
      <td>0.0488</td>
      <td>0.0103</td>
      <td>0.000000</td>
      <td>0.612</td>
      <td>0.684</td>
      <td>148.726</td>
      <td>rock, pop</td>
      <td>[rock,, pop]</td>
    </tr>
  </tbody>
</table>
</div>




```python
# let's describe the data, and pivot the table with .T
spotify_df.describe().T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>duration_ms</th>
      <td>2000.0</td>
      <td>228748.124500</td>
      <td>39136.569008</td>
      <td>113000.000000</td>
      <td>203580.00000</td>
      <td>223279.50000</td>
      <td>248133.000000</td>
      <td>484146.000</td>
    </tr>
    <tr>
      <th>year</th>
      <td>2000.0</td>
      <td>2009.494000</td>
      <td>5.859960</td>
      <td>1998.000000</td>
      <td>2004.00000</td>
      <td>2010.00000</td>
      <td>2015.000000</td>
      <td>2020.000</td>
    </tr>
    <tr>
      <th>popularity</th>
      <td>2000.0</td>
      <td>59.872500</td>
      <td>21.335577</td>
      <td>0.000000</td>
      <td>56.00000</td>
      <td>65.50000</td>
      <td>73.000000</td>
      <td>89.000</td>
    </tr>
    <tr>
      <th>danceability</th>
      <td>2000.0</td>
      <td>0.667437</td>
      <td>0.140416</td>
      <td>0.129000</td>
      <td>0.58100</td>
      <td>0.67600</td>
      <td>0.764000</td>
      <td>0.975</td>
    </tr>
    <tr>
      <th>energy</th>
      <td>2000.0</td>
      <td>0.720366</td>
      <td>0.152745</td>
      <td>0.054900</td>
      <td>0.62200</td>
      <td>0.73600</td>
      <td>0.839000</td>
      <td>0.999</td>
    </tr>
    <tr>
      <th>key</th>
      <td>2000.0</td>
      <td>5.378000</td>
      <td>3.615059</td>
      <td>0.000000</td>
      <td>2.00000</td>
      <td>6.00000</td>
      <td>8.000000</td>
      <td>11.000</td>
    </tr>
    <tr>
      <th>loudness</th>
      <td>2000.0</td>
      <td>-5.512435</td>
      <td>1.933482</td>
      <td>-20.514000</td>
      <td>-6.49025</td>
      <td>-5.28500</td>
      <td>-4.167750</td>
      <td>-0.276</td>
    </tr>
    <tr>
      <th>mode</th>
      <td>2000.0</td>
      <td>0.553500</td>
      <td>0.497254</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>1.00000</td>
      <td>1.000000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <th>speechiness</th>
      <td>2000.0</td>
      <td>0.103568</td>
      <td>0.096159</td>
      <td>0.023200</td>
      <td>0.03960</td>
      <td>0.05985</td>
      <td>0.129000</td>
      <td>0.576</td>
    </tr>
    <tr>
      <th>acousticness</th>
      <td>2000.0</td>
      <td>0.128955</td>
      <td>0.173346</td>
      <td>0.000019</td>
      <td>0.01400</td>
      <td>0.05570</td>
      <td>0.176250</td>
      <td>0.976</td>
    </tr>
    <tr>
      <th>instrumentalness</th>
      <td>2000.0</td>
      <td>0.015226</td>
      <td>0.087771</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.000068</td>
      <td>0.985</td>
    </tr>
    <tr>
      <th>liveness</th>
      <td>2000.0</td>
      <td>0.181216</td>
      <td>0.140669</td>
      <td>0.021500</td>
      <td>0.08810</td>
      <td>0.12400</td>
      <td>0.241000</td>
      <td>0.853</td>
    </tr>
    <tr>
      <th>valence</th>
      <td>2000.0</td>
      <td>0.551690</td>
      <td>0.220864</td>
      <td>0.038100</td>
      <td>0.38675</td>
      <td>0.55750</td>
      <td>0.730000</td>
      <td>0.973</td>
    </tr>
    <tr>
      <th>tempo</th>
      <td>2000.0</td>
      <td>120.122558</td>
      <td>26.967112</td>
      <td>60.019000</td>
      <td>98.98575</td>
      <td>120.02150</td>
      <td>134.265500</td>
      <td>210.851</td>
    </tr>
  </tbody>
</table>
</div>




```python
# let's plot histograms for each variable to see what the distributions look like
spotify_df.hist(bins=50, figsize=(20,15))
plt.show()
```


    
![png](output_7_0.png)
    



```python
# here we'll convert the popularity into bins for popular/not popular
spotify_df["popularity_binned"] = pd.cut(spotify_df["popularity"],
                              bins = [0, 62, 90],
                              labels = [0, 1], include_lowest=True)
```


```python
# let's show the distribution of pop vs not pop
spotify_df["popularity_binned"].value_counts(dropna=False, normalize=True)*100
```




    1    59.3
    0    40.7
    Name: popularity_binned, dtype: float64




```python
# I originally had an entire section for deleting duplicates, however this broke the SSS due to indexing issues. 
# these have since been removed and this started working again.

split = StratifiedShuffleSplit(n_splits=1 ,test_size=0.2, random_state=9001)

for train_index, test_index in split.split(spotify_df, spotify_df["popularity_binned"]):
    split_train = spotify_df.loc[train_index]
    split_test = spotify_df.loc[test_index]
```


```python
# let's check the skew and kurtosis for the data
skew = pd.Series(split_train.skew(),name="skew")
kurtosis = pd.Series(split_train.kurtosis(),name="kurtosis")
pd.concat([skew,kurtosis],axis =1)
```

    C:\Users\MYDICK~1\AppData\Local\Temp/ipykernel_44148/1428877543.py:2: FutureWarning: Dropping of nuisance columns in DataFrame reductions (with 'numeric_only=None') is deprecated; in a future version this will raise TypeError.  Select only valid columns before calling the reduction.
      skew = pd.Series(split_train.skew(),name="skew")
    C:\Users\MYDICK~1\AppData\Local\Temp/ipykernel_44148/1428877543.py:3: FutureWarning: Dropping of nuisance columns in DataFrame reductions (with 'numeric_only=None') is deprecated; in a future version this will raise TypeError.  Select only valid columns before calling the reduction.
      kurtosis = pd.Series(split_train.kurtosis(),name="kurtosis")
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>skew</th>
      <th>kurtosis</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>duration_ms</th>
      <td>1.018902</td>
      <td>3.265376</td>
    </tr>
    <tr>
      <th>explicit</th>
      <td>1.022867</td>
      <td>-0.954939</td>
    </tr>
    <tr>
      <th>year</th>
      <td>-0.028587</td>
      <td>-1.200091</td>
    </tr>
    <tr>
      <th>popularity</th>
      <td>-1.845393</td>
      <td>2.757777</td>
    </tr>
    <tr>
      <th>danceability</th>
      <td>-0.459228</td>
      <td>0.159500</td>
    </tr>
    <tr>
      <th>energy</th>
      <td>-0.656024</td>
      <td>0.312530</td>
    </tr>
    <tr>
      <th>key</th>
      <td>-0.012106</td>
      <td>-1.288990</td>
    </tr>
    <tr>
      <th>loudness</th>
      <td>-1.262497</td>
      <td>4.363057</td>
    </tr>
    <tr>
      <th>mode</th>
      <td>-0.206279</td>
      <td>-1.959900</td>
    </tr>
    <tr>
      <th>speechiness</th>
      <td>1.774119</td>
      <td>2.687664</td>
    </tr>
    <tr>
      <th>acousticness</th>
      <td>2.119104</td>
      <td>4.786895</td>
    </tr>
    <tr>
      <th>instrumentalness</th>
      <td>7.261777</td>
      <td>56.163529</td>
    </tr>
    <tr>
      <th>liveness</th>
      <td>1.914616</td>
      <td>4.149754</td>
    </tr>
    <tr>
      <th>valence</th>
      <td>-0.133275</td>
      <td>-0.842183</td>
    </tr>
    <tr>
      <th>tempo</th>
      <td>0.601136</td>
      <td>0.134137</td>
    </tr>
  </tbody>
</table>
</div>



Speechiness, loudness, instrumentalness and liveness are skewed. While instrumentalness has quite a high kurtosis. StandardScaler() may be helpful to bring this all in line with each other.


```python
# let's go through the columns and start processing the data
# .head to get the column names faster
split_train.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>artist</th>
      <th>song</th>
      <th>duration_ms</th>
      <th>explicit</th>
      <th>year</th>
      <th>popularity</th>
      <th>danceability</th>
      <th>energy</th>
      <th>key</th>
      <th>loudness</th>
      <th>mode</th>
      <th>speechiness</th>
      <th>acousticness</th>
      <th>instrumentalness</th>
      <th>liveness</th>
      <th>valence</th>
      <th>tempo</th>
      <th>genre</th>
      <th>genre_list</th>
      <th>popularity_binned</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1869</th>
      <td>Drake</td>
      <td>In My Feelings</td>
      <td>217925</td>
      <td>True</td>
      <td>2018</td>
      <td>75</td>
      <td>0.835</td>
      <td>0.626</td>
      <td>1</td>
      <td>-5.833</td>
      <td>1</td>
      <td>0.1250</td>
      <td>0.0589</td>
      <td>0.00006</td>
      <td>0.396</td>
      <td>0.350</td>
      <td>91.030</td>
      <td>hip hop, pop, R&amp;B</td>
      <td>[hip, hop,, pop,, R&amp;B]</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1511</th>
      <td>The Weeknd</td>
      <td>The Hills</td>
      <td>242253</td>
      <td>True</td>
      <td>2015</td>
      <td>84</td>
      <td>0.585</td>
      <td>0.564</td>
      <td>0</td>
      <td>-7.063</td>
      <td>0</td>
      <td>0.0515</td>
      <td>0.0671</td>
      <td>0.00000</td>
      <td>0.135</td>
      <td>0.137</td>
      <td>113.003</td>
      <td>pop, R&amp;B</td>
      <td>[pop,, R&amp;B]</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
split_train['duration_ms'].describe()
# duration is important, however ms is a tad unwieldy. We should convert this column into full seconds
```




    count      1600.000000
    mean     229066.065625
    std       39484.854532
    min      114893.000000
    25%      203502.750000
    50%      223471.500000
    75%      248639.500000
    max      484146.000000
    Name: duration_ms, dtype: float64




```python
split_train['explicit'].describe()
# this is a t/f boolean. Feeding this into a model, this might be best converted to a 1/0. However, I don't think it will be
# necessary to.
```




    count      1600
    unique        2
    top       False
    freq       1164
    Name: explicit, dtype: object




```python
split_train['year'].value_counts
# year column doesn't seem to be too relevant. However, to limit the amount of options let's bin this
```




    <bound method IndexOpsMixin.value_counts of 1869    2018
    1511    2015
    1743    2017
    201     2002
    856     2008
            ... 
    1406    2014
    67      2000
    482     2003
    1530    2014
    300     2003
    Name: year, Length: 1600, dtype: int64>




```python
# let's take the release year and subtract it from the current year in both train and test sets
split_train['age'] = 2022 - split_train['year'] 
split_test['age'] = 2022 - split_test['year'] 

# now we'll bin them in both to reduce the variability
split_train["age_binned"]=pd.cut(split_train["age"],
                              bins = [0, 5, 15, 30],
                              labels = [1, 2, 3], include_lowest = True)

split_test["age_binned"]=pd.cut(split_test["age"],
                              bins = [0, 5, 15, 30],
                              labels = [1, 2, 3],include_lowest = True)
```


```python
# Danceability - A value of 0.0 is least danceable and 1.0 is most danceable.
split_train['danceability'].describe()

# this variable is already fairly preprocessed. This won't need any work from me.
```




    count    1600.000000
    mean        0.671513
    std         0.141112
    min         0.179000
    25%         0.585000
    50%         0.680000
    75%         0.771000
    max         0.975000
    Name: danceability, dtype: float64




```python
# Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity.
split_train['energy'].describe()

# this variable is already fairly preprocessed. This won't need any work from me.
```




    count    1600.000000
    mean        0.721387
    std         0.151418
    min         0.054900
    25%         0.626000
    50%         0.736000
    75%         0.841000
    max         0.999000
    Name: energy, dtype: float64




```python
# Key - he key the track is in. Integers map to pitches using standard Pitch Class notation. E.g. 0 = C, 1 = C♯/D♭, 2 = D, and so on. 
split_train['key'].describe()

# song keys are typically categorical. This column will need to be altered a bit to take this into account
```




    count    1600.000000
    mean        5.363750
    std         3.612422
    min         0.000000
    25%         2.000000
    50%         6.000000
    75%         8.000000
    max        11.000000
    Name: key, dtype: float64




```python
# Loudness - Values typically range between -60 and 0 db. 
split_train['loudness'].describe()

# this variable is already fairly preprocessed. This won't need any work from me.
```




    count    1600.000000
    mean       -5.503811
    std         1.927214
    min       -20.514000
    25%        -6.479000
    50%        -5.263500
    75%        -4.167000
    max        -0.276000
    Name: loudness, dtype: float64




```python
# Feature 8: mode: Mode indicates the modality (major or minor) of a track,
# the type of scale from which its melodic content is derived. Major is represented by 1 and minor is 0.
split_train['mode'].value_counts()

# mode here is being used similarly to a t/f boolean, however is being shown as maj/min boolean.
# This won't need any work from me.
```




    1    882
    0    718
    Name: mode, dtype: int64




```python
# Speechiness detects the presence of spoken words in a track. The more exclusively speech-like 
# the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value. Values 
# above 0.66 describe tracks that are probably made entirely of spoken words. 
# Values between 0.33 and 0.66 describe tracks that may contain both music and speech, either in sections or layered, 
# including such cases as rap music. Values below 0.33 most likely represent music and other non-speech-like tracks.
split_train['speechiness'].describe()

# this variable is already fairly preprocessed. This won't need any work from me.
```




    count    1600.000000
    mean        0.103749
    std         0.096478
    min         0.023200
    25%         0.039850
    50%         0.060000
    75%         0.128000
    max         0.576000
    Name: speechiness, dtype: float64




```python
# Acousticness: A confidence measure from 0.0 to 1.0 
# of whether the track is acoustic. 1.0 represents high confidence the track is acoustic.
split_train['acousticness'].describe()

# this variable is already fairly preprocessed. This won't need any work from me.
```




    count    1600.000000
    mean        0.126943
    std         0.171726
    min         0.000019
    25%         0.013475
    50%         0.054750
    75%         0.173250
    max         0.976000
    Name: acousticness, dtype: float64




```python
# Instrumentalness: Predicts whether a track contains no vocals. "Ooh" and "aah" sounds are treated as instrumental in this 
# context. Rap or spoken word tracks are clearly "vocal". The closer the instrumentalness value is to 1.0, the greater 
# likelihood the track contains no vocal content. Values above 0.5 are intended to represent instrumental tracks, but
# confidence is higher as the value approaches 1.0
split_train['instrumentalness'].describe()

# this variable is already fairly preprocessed. However, the skewness and kurtosis of this column indicates that a standard
# scaler might help make this easier to work with. (75% percentile is only .000062, however max is .985000)
```




    count    1600.000000
    mean        0.016573
    std         0.092426
    min         0.000000
    25%         0.000000
    50%         0.000000
    75%         0.000071
    max         0.985000
    Name: instrumentalness, dtype: float64




```python
# liveness: Detects the presence of an audience in the recording. Higher liveness values represent an increased probability
# that the track was performed live. A value above 0.8 provides strong likelihood that the track is live.
split_train['liveness'].describe()
```




    count    1600.000000
    mean        0.179024
    std         0.140756
    min         0.021500
    25%         0.086775
    50%         0.124000
    75%         0.233250
    max         0.853000
    Name: liveness, dtype: float64




```python
# Valence: A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high
# valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative
# (e.g. sad, depressed, angry).
split_train['valence'].describe()

# this variable is already fairly preprocessed. This won't need any work from me.
```




    count    1600.000000
    mean        0.555079
    std         0.223600
    min         0.038100
    25%         0.387000
    50%         0.560500
    75%         0.738000
    max         0.973000
    Name: valence, dtype: float64




```python
# tempo: The overall estimated tempo of a track in beats per minute (BPM). In musical terminology, 
# tempo is the speed or pace of a given piece and derives directly from the average beat duration.
split_train['tempo'].describe()

# this variable is already fairly preprocessed. This won't need any work from me.
```




    count    1600.000000
    mean      119.565425
    std        26.890377
    min        60.019000
    25%        98.060000
    50%       119.992000
    75%       133.592000
    max       210.851000
    Name: tempo, dtype: float64



We've cleaned up the genres, and decided a standard scaler will be useful for instrumentalism. Converting ms might be useful.
Now let's look at some correlation matrices


```python
correlation_matrix_spotify = split_train.corr()

# let's grab the popularity column to see if there's any strong correlations to popularity from other variables
correlation_matrix_spotify.popularity

# doesn't appear to be any strong correlations here.
```




    duration_ms         0.054830
    explicit            0.080530
    year               -0.022828
    popularity          1.000000
    danceability        0.006018
    energy             -0.016046
    key                 0.034523
    loudness            0.039392
    mode               -0.024705
    speechiness         0.038875
    acousticness        0.019229
    instrumentalness   -0.063858
    liveness           -0.012743
    valence            -0.029968
    tempo              -0.003411
    age                 0.022828
    Name: popularity, dtype: float64




```python
# let's look to see how popularity compares to the key of the song
sns.catplot(x="key", y="popularity", kind="box", data=spotify_df)

# here we can see that key 3 (D#) has a tighter popularity, where as key 4 (E) has a much wider spread with more outliers
```




    <seaborn.axisgrid.FacetGrid at 0x1f7fb6eba60>




    
![png](output_31_1.png)
    



```python
sns.catplot(x="explicit", y="popularity", kind="box", data=spotify_df)

# this leads to me believe that explicit songs are more popular, due to the tighter spread on true.
# However, the overall spreads are similar enough that I can't see a real correlation between the explicity and popularity.
```




    <seaborn.axisgrid.FacetGrid at 0x1f7fa2c93d0>




    
![png](output_32_1.png)
    



```python
sns.catplot(x="genre", y="popularity", kind="box", data=spotify_df)
# well. That's an ugly chart. But we can see a very wide spread amongst all genres. Even if this is unreadable. I would
# most likely have to split things up by genre in order to find any real correlation between the genres themselves
```




    <seaborn.axisgrid.FacetGrid at 0x1f7fb5a5190>




    
![png](output_33_1.png)
    



```python
# let's clean up the text columns

# grab the stopwords from nltk
STOPWORDS = set(stopwords.words('english'))

# grab the punctuation from the string library
PUNCT_TO_REMOVE = string.punctuation

# let's get rid of those punctuations
def remove_punctuation(x):
    return x.translate(str.maketrans('', '', PUNCT_TO_REMOVE))

# clean the text with pounctuation remover
def text_clean(x):
    split_train[x]= split_train[x].str.lower() # lower case all the text for easier parsing
    split_train[x] = split_train[x].apply(lambda x: remove_punctuation(x)) # delete the punctuation
    
# just remove the stop words from the nltk library
def remove_stopwords(x):
    return " ".join([word for word in str(x).split() if word not in STOPWORDS])


# run the functions based on column name
text_clean('genre')
text_clean('song')
text_clean('artist')

# run the functions for both train and test sets
split_train['genre'] = split_train['genre'].apply(lambda x: remove_stopwords(x))
split_train['song'] = split_train['song'].apply(lambda x: remove_stopwords(x))
split_train['artist'] = split_train['artist'].apply(lambda x: remove_stopwords(x))

split_test['genre'] = split_test['genre'].apply(lambda x: remove_stopwords(x))
split_test['song'] = split_test['song'].apply(lambda x: remove_stopwords(x))
split_test['artist'] = split_test['artist'].apply(lambda x: remove_stopwords(x))
```


```python
# let's use PorterStemmer to normalize some of the slang words
stemmer = PorterStemmer()
def stem_words(text):
    return " ".join([stemmer.stem(word) for word in text.split()])

# apply to both test and train
split_train["genre"] = split_train["genre"].apply(lambda text: stem_words(text))
split_train["song"] = split_train["song"].apply(lambda text: stem_words(text))

split_test["genre"] = split_test["genre"].apply(lambda text: stem_words(text))
split_test["song"] = split_test["song"].apply(lambda text: stem_words(text))
```


```python
# lemmatizer merges inflected words with their real words for easier processing
lemmatizer = WordNetLemmatizer()

# map the words to their identifiers for easier processing
wordnet_map = {"N":wordnet.NOUN, "V":wordnet.VERB, "J":wordnet.ADJ, "R":wordnet.ADV}

# function to run the lematizer based on the map
def lemmatize_words(text):
    pos_tagged_text = nltk.pos_tag(text.split())
    return " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text])


# apply to both sets
split_train["genre"] = split_train["genre"].apply(lambda text: lemmatize_words(text))
split_train["song"] = split_train["song"].apply(lambda text: lemmatize_words(text))

split_test["genre"] = split_test["genre"].apply(lambda text: lemmatize_words(text))
split_test["song"] = split_test["song"].apply(lambda text: lemmatize_words(text))
```


```python
# let's create an object to hold all the column names that are numerical
all_no_text = ['duration_ms', 'explicit', 'year', 'danceability', 'energy', 'key', 'loudness', 'mode', 
               'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'popularity_binned', 'age_binned']

all_w_text = ['artist', 'song','genre', 'duration_ms', 'explicit', 'year',  'danceability',
              'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness',
              'valence', 'tempo', 'popularity_binned', 'age', 'age_binned']
```


```python
# let's make the train and test set and get everything sorted
X_train = split_train[all_no_text]
Y_train = split_train['popularity_binned']

X_test = split_test[all_no_text]
Y_test = split_test['popularity_binned']

X_train_text = split_train[all_w_text]
X_test_text = split_test[all_w_text]
```


```python
# let's set up the encoders and scalers
ohe = OneHotEncoder(handle_unknown='ignore')
oe = OrdinalEncoder()
scaler = StandardScaler()
vect = CountVectorizer()

logreg = LogisticRegression(solver='liblinear', multi_class='auto', random_state=9001, max_iter=1000)
```


```python
# Let's make the column transformer
ct = make_column_transformer(
    (ohe, ['explicit', 'key']),
    (oe,['age_binned']),
    (scaler, ['duration_ms', 'danceability', 'energy', 'loudness', 'mode',
              'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']), remainder='drop')
```


```python
# Pipeline for Logistic Regression Classification problem

# here we make the pipeline, and pass in nthe transformer before applying the logreg
pipe_LR = make_pipeline(ct, logreg)

# now we fit the logreg
pipe_LR.fit(X_train, Y_train)
```




<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>Pipeline(steps=[(&#x27;columntransformer&#x27;,
                 ColumnTransformer(transformers=[(&#x27;onehotencoder&#x27;,
                                                  OneHotEncoder(handle_unknown=&#x27;ignore&#x27;),
                                                  [&#x27;explicit&#x27;, &#x27;key&#x27;]),
                                                 (&#x27;ordinalencoder&#x27;,
                                                  OrdinalEncoder(),
                                                  [&#x27;age_binned&#x27;]),
                                                 (&#x27;standardscaler&#x27;,
                                                  StandardScaler(),
                                                  [&#x27;duration_ms&#x27;,
                                                   &#x27;danceability&#x27;, &#x27;energy&#x27;,
                                                   &#x27;loudness&#x27;, &#x27;mode&#x27;,
                                                   &#x27;speechiness&#x27;,
                                                   &#x27;acousticness&#x27;,
                                                   &#x27;instrumentalness&#x27;,
                                                   &#x27;liveness&#x27;, &#x27;valence&#x27;,
                                                   &#x27;tempo&#x27;])])),
                (&#x27;logisticregression&#x27;,
                 LogisticRegression(max_iter=1000, random_state=9001,
                                    solver=&#x27;liblinear&#x27;))])</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" ><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">Pipeline</label><div class="sk-toggleable__content"><pre>Pipeline(steps=[(&#x27;columntransformer&#x27;,
                 ColumnTransformer(transformers=[(&#x27;onehotencoder&#x27;,
                                                  OneHotEncoder(handle_unknown=&#x27;ignore&#x27;),
                                                  [&#x27;explicit&#x27;, &#x27;key&#x27;]),
                                                 (&#x27;ordinalencoder&#x27;,
                                                  OrdinalEncoder(),
                                                  [&#x27;age_binned&#x27;]),
                                                 (&#x27;standardscaler&#x27;,
                                                  StandardScaler(),
                                                  [&#x27;duration_ms&#x27;,
                                                   &#x27;danceability&#x27;, &#x27;energy&#x27;,
                                                   &#x27;loudness&#x27;, &#x27;mode&#x27;,
                                                   &#x27;speechiness&#x27;,
                                                   &#x27;acousticness&#x27;,
                                                   &#x27;instrumentalness&#x27;,
                                                   &#x27;liveness&#x27;, &#x27;valence&#x27;,
                                                   &#x27;tempo&#x27;])])),
                (&#x27;logisticregression&#x27;,
                 LogisticRegression(max_iter=1000, random_state=9001,
                                    solver=&#x27;liblinear&#x27;))])</pre></div></div></div><div class="sk-serial"><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" ><label for="sk-estimator-id-2" class="sk-toggleable__label sk-toggleable__label-arrow">columntransformer: ColumnTransformer</label><div class="sk-toggleable__content"><pre>ColumnTransformer(transformers=[(&#x27;onehotencoder&#x27;,
                                 OneHotEncoder(handle_unknown=&#x27;ignore&#x27;),
                                 [&#x27;explicit&#x27;, &#x27;key&#x27;]),
                                (&#x27;ordinalencoder&#x27;, OrdinalEncoder(),
                                 [&#x27;age_binned&#x27;]),
                                (&#x27;standardscaler&#x27;, StandardScaler(),
                                 [&#x27;duration_ms&#x27;, &#x27;danceability&#x27;, &#x27;energy&#x27;,
                                  &#x27;loudness&#x27;, &#x27;mode&#x27;, &#x27;speechiness&#x27;,
                                  &#x27;acousticness&#x27;, &#x27;instrumentalness&#x27;,
                                  &#x27;liveness&#x27;, &#x27;valence&#x27;, &#x27;tempo&#x27;])])</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-3" type="checkbox" ><label for="sk-estimator-id-3" class="sk-toggleable__label sk-toggleable__label-arrow">onehotencoder</label><div class="sk-toggleable__content"><pre>[&#x27;explicit&#x27;, &#x27;key&#x27;]</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-4" type="checkbox" ><label for="sk-estimator-id-4" class="sk-toggleable__label sk-toggleable__label-arrow">OneHotEncoder</label><div class="sk-toggleable__content"><pre>OneHotEncoder(handle_unknown=&#x27;ignore&#x27;)</pre></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-5" type="checkbox" ><label for="sk-estimator-id-5" class="sk-toggleable__label sk-toggleable__label-arrow">ordinalencoder</label><div class="sk-toggleable__content"><pre>[&#x27;age_binned&#x27;]</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-6" type="checkbox" ><label for="sk-estimator-id-6" class="sk-toggleable__label sk-toggleable__label-arrow">OrdinalEncoder</label><div class="sk-toggleable__content"><pre>OrdinalEncoder()</pre></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-7" type="checkbox" ><label for="sk-estimator-id-7" class="sk-toggleable__label sk-toggleable__label-arrow">standardscaler</label><div class="sk-toggleable__content"><pre>[&#x27;duration_ms&#x27;, &#x27;danceability&#x27;, &#x27;energy&#x27;, &#x27;loudness&#x27;, &#x27;mode&#x27;, &#x27;speechiness&#x27;, &#x27;acousticness&#x27;, &#x27;instrumentalness&#x27;, &#x27;liveness&#x27;, &#x27;valence&#x27;, &#x27;tempo&#x27;]</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-8" type="checkbox" ><label for="sk-estimator-id-8" class="sk-toggleable__label sk-toggleable__label-arrow">StandardScaler</label><div class="sk-toggleable__content"><pre>StandardScaler()</pre></div></div></div></div></div></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-9" type="checkbox" ><label for="sk-estimator-id-9" class="sk-toggleable__label sk-toggleable__label-arrow">LogisticRegression</label><div class="sk-toggleable__content"><pre>LogisticRegression(max_iter=1000, random_state=9001, solver=&#x27;liblinear&#x27;)</pre></div></div></div></div></div></div></div>




```python
# let's get the cross validation score after a few repititions
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=9001)

cross_val_score(pipe_LR, X_train, Y_train, cv=cv, scoring='accuracy').mean()
```




    0.6151875




```python
# let's apply logreg params and see which regression works best
params = {}
params['logisticregression__penalty'] = ['l1', 'l2']
params['logisticregression__C'] = [0.1, 1, 10]
```


```python
# let's apply the model and params to see average responses and how long the models take to run
grid = GridSearchCV(pipe_LR, params, cv=cv, scoring='accuracy')
grid.fit(X_train, Y_train);
```


```python
# now let's make a df of the results and desplay them.
results = pd.DataFrame(grid.cv_results_)
# let's sort them for easier readability
results.sort_values('rank_test_score')

# based off the sort, I can already tell the best avg score is .618125, using penalty l1 and C of 1
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean_fit_time</th>
      <th>std_fit_time</th>
      <th>mean_score_time</th>
      <th>std_score_time</th>
      <th>param_logisticregression__C</th>
      <th>param_logisticregression__penalty</th>
      <th>params</th>
      <th>split0_test_score</th>
      <th>split1_test_score</th>
      <th>split2_test_score</th>
      <th>split3_test_score</th>
      <th>split4_test_score</th>
      <th>...</th>
      <th>split91_test_score</th>
      <th>split92_test_score</th>
      <th>split93_test_score</th>
      <th>split94_test_score</th>
      <th>split95_test_score</th>
      <th>split96_test_score</th>
      <th>split97_test_score</th>
      <th>split98_test_score</th>
      <th>split99_test_score</th>
      <th>mean_test_score</th>
      <th>std_test_score</th>
      <th>rank_test_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>0.017367</td>
      <td>0.003080</td>
      <td>0.003899</td>
      <td>0.000856</td>
      <td>1</td>
      <td>l1</td>
      <td>{'logisticregression__C': 1, 'logisticregressi...</td>
      <td>0.64375</td>
      <td>0.61875</td>
      <td>0.57500</td>
      <td>0.63125</td>
      <td>0.64375</td>
      <td>...</td>
      <td>0.55625</td>
      <td>0.61875</td>
      <td>0.64375</td>
      <td>0.60000</td>
      <td>0.62500</td>
      <td>0.63750</td>
      <td>0.65000</td>
      <td>0.60000</td>
      <td>0.60625</td>
      <td>0.617688</td>
      <td>0.028077</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.009243</td>
      <td>0.002263</td>
      <td>0.004081</td>
      <td>0.001079</td>
      <td>0.1</td>
      <td>l2</td>
      <td>{'logisticregression__C': 0.1, 'logisticregres...</td>
      <td>0.64375</td>
      <td>0.61875</td>
      <td>0.57500</td>
      <td>0.63125</td>
      <td>0.62500</td>
      <td>...</td>
      <td>0.56875</td>
      <td>0.61875</td>
      <td>0.62500</td>
      <td>0.57500</td>
      <td>0.61875</td>
      <td>0.64375</td>
      <td>0.63750</td>
      <td>0.61250</td>
      <td>0.60625</td>
      <td>0.615250</td>
      <td>0.027334</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.008278</td>
      <td>0.000403</td>
      <td>0.003505</td>
      <td>0.000265</td>
      <td>1</td>
      <td>l2</td>
      <td>{'logisticregression__C': 1, 'logisticregressi...</td>
      <td>0.62500</td>
      <td>0.61875</td>
      <td>0.57500</td>
      <td>0.63125</td>
      <td>0.63750</td>
      <td>...</td>
      <td>0.55625</td>
      <td>0.62500</td>
      <td>0.65000</td>
      <td>0.58125</td>
      <td>0.61875</td>
      <td>0.62500</td>
      <td>0.63750</td>
      <td>0.61875</td>
      <td>0.60625</td>
      <td>0.615187</td>
      <td>0.027290</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.091709</td>
      <td>0.009417</td>
      <td>0.003758</td>
      <td>0.000506</td>
      <td>10</td>
      <td>l1</td>
      <td>{'logisticregression__C': 10, 'logisticregress...</td>
      <td>0.62500</td>
      <td>0.62500</td>
      <td>0.57500</td>
      <td>0.63125</td>
      <td>0.63125</td>
      <td>...</td>
      <td>0.55000</td>
      <td>0.62500</td>
      <td>0.65000</td>
      <td>0.58750</td>
      <td>0.61875</td>
      <td>0.62500</td>
      <td>0.63750</td>
      <td>0.61250</td>
      <td>0.60625</td>
      <td>0.615000</td>
      <td>0.027979</td>
      <td>4</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0.009195</td>
      <td>0.000805</td>
      <td>0.003729</td>
      <td>0.000389</td>
      <td>0.1</td>
      <td>l1</td>
      <td>{'logisticregression__C': 0.1, 'logisticregres...</td>
      <td>0.65000</td>
      <td>0.59375</td>
      <td>0.58125</td>
      <td>0.61875</td>
      <td>0.62500</td>
      <td>...</td>
      <td>0.59375</td>
      <td>0.60625</td>
      <td>0.63750</td>
      <td>0.61875</td>
      <td>0.61250</td>
      <td>0.63750</td>
      <td>0.62500</td>
      <td>0.60000</td>
      <td>0.60625</td>
      <td>0.614875</td>
      <td>0.024777</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.008718</td>
      <td>0.001011</td>
      <td>0.003777</td>
      <td>0.000606</td>
      <td>10</td>
      <td>l2</td>
      <td>{'logisticregression__C': 10, 'logisticregress...</td>
      <td>0.62500</td>
      <td>0.61875</td>
      <td>0.57500</td>
      <td>0.63125</td>
      <td>0.63750</td>
      <td>...</td>
      <td>0.55000</td>
      <td>0.63125</td>
      <td>0.65625</td>
      <td>0.58750</td>
      <td>0.61875</td>
      <td>0.62500</td>
      <td>0.63125</td>
      <td>0.61250</td>
      <td>0.60625</td>
      <td>0.614688</td>
      <td>0.027858</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
<p>6 rows × 110 columns</p>
</div>




```python
# using the best_score_ function, we can confirm that .618125 was the best score
grid.best_score_
```




    0.6176875000000002




```python
# and best_params_ will grab the C and penalty that led to that score
grid.best_params_
```




    {'logisticregression__C': 1, 'logisticregression__penalty': 'l1'}




```python
# now let's see how the model works with text instead of just numerical values
ct_w_text = make_column_transformer(
    (ohe, ['explicit', 'key']),
    (vect, 'genre','artist','song'),
    (TfidfVectorizer(), 'genre','artist','song'),
    (oe,['age_binned']),
    (scaler, ['duration_ms', 'danceability', 'energy', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']),
    remainder='drop')
```


```python
# using tfidf and countvectorizer, we can create another pipeline
pipe_LR = make_pipeline(ct_w_text, logreg)
pipe_LR.fit(X_train_text, Y_train)
```




<style>#sk-container-id-2 {color: black;background-color: white;}#sk-container-id-2 pre{padding: 0;}#sk-container-id-2 div.sk-toggleable {background-color: white;}#sk-container-id-2 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-2 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-2 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-2 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-2 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-2 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-2 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-2 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-2 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-2 div.sk-item {position: relative;z-index: 1;}#sk-container-id-2 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-2 div.sk-item::before, #sk-container-id-2 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-2 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-2 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-2 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-2 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-2 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-2 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-2 div.sk-label-container {text-align: center;}#sk-container-id-2 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-2 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-2" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>Pipeline(steps=[(&#x27;columntransformer&#x27;,
                 ColumnTransformer(transformers=[(&#x27;onehotencoder&#x27;,
                                                  OneHotEncoder(handle_unknown=&#x27;ignore&#x27;),
                                                  [&#x27;explicit&#x27;, &#x27;key&#x27;]),
                                                 (&#x27;countvectorizer&#x27;,
                                                  CountVectorizer(), &#x27;genre&#x27;),
                                                 (&#x27;tfidfvectorizer&#x27;,
                                                  TfidfVectorizer(), &#x27;genre&#x27;),
                                                 (&#x27;ordinalencoder&#x27;,
                                                  OrdinalEncoder(),
                                                  [&#x27;age_binned&#x27;]),
                                                 (&#x27;standardscaler&#x27;,
                                                  StandardScaler(),
                                                  [&#x27;duration_ms&#x27;,
                                                   &#x27;danceability&#x27;, &#x27;energy&#x27;,
                                                   &#x27;loudness&#x27;, &#x27;mode&#x27;,
                                                   &#x27;speechiness&#x27;,
                                                   &#x27;acousticness&#x27;,
                                                   &#x27;instrumentalness&#x27;,
                                                   &#x27;liveness&#x27;, &#x27;valence&#x27;,
                                                   &#x27;tempo&#x27;])])),
                (&#x27;logisticregression&#x27;,
                 LogisticRegression(max_iter=1000, random_state=9001,
                                    solver=&#x27;liblinear&#x27;))])</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-10" type="checkbox" ><label for="sk-estimator-id-10" class="sk-toggleable__label sk-toggleable__label-arrow">Pipeline</label><div class="sk-toggleable__content"><pre>Pipeline(steps=[(&#x27;columntransformer&#x27;,
                 ColumnTransformer(transformers=[(&#x27;onehotencoder&#x27;,
                                                  OneHotEncoder(handle_unknown=&#x27;ignore&#x27;),
                                                  [&#x27;explicit&#x27;, &#x27;key&#x27;]),
                                                 (&#x27;countvectorizer&#x27;,
                                                  CountVectorizer(), &#x27;genre&#x27;),
                                                 (&#x27;tfidfvectorizer&#x27;,
                                                  TfidfVectorizer(), &#x27;genre&#x27;),
                                                 (&#x27;ordinalencoder&#x27;,
                                                  OrdinalEncoder(),
                                                  [&#x27;age_binned&#x27;]),
                                                 (&#x27;standardscaler&#x27;,
                                                  StandardScaler(),
                                                  [&#x27;duration_ms&#x27;,
                                                   &#x27;danceability&#x27;, &#x27;energy&#x27;,
                                                   &#x27;loudness&#x27;, &#x27;mode&#x27;,
                                                   &#x27;speechiness&#x27;,
                                                   &#x27;acousticness&#x27;,
                                                   &#x27;instrumentalness&#x27;,
                                                   &#x27;liveness&#x27;, &#x27;valence&#x27;,
                                                   &#x27;tempo&#x27;])])),
                (&#x27;logisticregression&#x27;,
                 LogisticRegression(max_iter=1000, random_state=9001,
                                    solver=&#x27;liblinear&#x27;))])</pre></div></div></div><div class="sk-serial"><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-11" type="checkbox" ><label for="sk-estimator-id-11" class="sk-toggleable__label sk-toggleable__label-arrow">columntransformer: ColumnTransformer</label><div class="sk-toggleable__content"><pre>ColumnTransformer(transformers=[(&#x27;onehotencoder&#x27;,
                                 OneHotEncoder(handle_unknown=&#x27;ignore&#x27;),
                                 [&#x27;explicit&#x27;, &#x27;key&#x27;]),
                                (&#x27;countvectorizer&#x27;, CountVectorizer(), &#x27;genre&#x27;),
                                (&#x27;tfidfvectorizer&#x27;, TfidfVectorizer(), &#x27;genre&#x27;),
                                (&#x27;ordinalencoder&#x27;, OrdinalEncoder(),
                                 [&#x27;age_binned&#x27;]),
                                (&#x27;standardscaler&#x27;, StandardScaler(),
                                 [&#x27;duration_ms&#x27;, &#x27;danceability&#x27;, &#x27;energy&#x27;,
                                  &#x27;loudness&#x27;, &#x27;mode&#x27;, &#x27;speechiness&#x27;,
                                  &#x27;acousticness&#x27;, &#x27;instrumentalness&#x27;,
                                  &#x27;liveness&#x27;, &#x27;valence&#x27;, &#x27;tempo&#x27;])])</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-12" type="checkbox" ><label for="sk-estimator-id-12" class="sk-toggleable__label sk-toggleable__label-arrow">onehotencoder</label><div class="sk-toggleable__content"><pre>[&#x27;explicit&#x27;, &#x27;key&#x27;]</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-13" type="checkbox" ><label for="sk-estimator-id-13" class="sk-toggleable__label sk-toggleable__label-arrow">OneHotEncoder</label><div class="sk-toggleable__content"><pre>OneHotEncoder(handle_unknown=&#x27;ignore&#x27;)</pre></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-14" type="checkbox" ><label for="sk-estimator-id-14" class="sk-toggleable__label sk-toggleable__label-arrow">countvectorizer</label><div class="sk-toggleable__content"><pre>genre</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-15" type="checkbox" ><label for="sk-estimator-id-15" class="sk-toggleable__label sk-toggleable__label-arrow">CountVectorizer</label><div class="sk-toggleable__content"><pre>CountVectorizer()</pre></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-16" type="checkbox" ><label for="sk-estimator-id-16" class="sk-toggleable__label sk-toggleable__label-arrow">tfidfvectorizer</label><div class="sk-toggleable__content"><pre>genre</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-17" type="checkbox" ><label for="sk-estimator-id-17" class="sk-toggleable__label sk-toggleable__label-arrow">TfidfVectorizer</label><div class="sk-toggleable__content"><pre>TfidfVectorizer()</pre></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-18" type="checkbox" ><label for="sk-estimator-id-18" class="sk-toggleable__label sk-toggleable__label-arrow">ordinalencoder</label><div class="sk-toggleable__content"><pre>[&#x27;age_binned&#x27;]</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-19" type="checkbox" ><label for="sk-estimator-id-19" class="sk-toggleable__label sk-toggleable__label-arrow">OrdinalEncoder</label><div class="sk-toggleable__content"><pre>OrdinalEncoder()</pre></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-20" type="checkbox" ><label for="sk-estimator-id-20" class="sk-toggleable__label sk-toggleable__label-arrow">standardscaler</label><div class="sk-toggleable__content"><pre>[&#x27;duration_ms&#x27;, &#x27;danceability&#x27;, &#x27;energy&#x27;, &#x27;loudness&#x27;, &#x27;mode&#x27;, &#x27;speechiness&#x27;, &#x27;acousticness&#x27;, &#x27;instrumentalness&#x27;, &#x27;liveness&#x27;, &#x27;valence&#x27;, &#x27;tempo&#x27;]</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-21" type="checkbox" ><label for="sk-estimator-id-21" class="sk-toggleable__label sk-toggleable__label-arrow">StandardScaler</label><div class="sk-toggleable__content"><pre>StandardScaler()</pre></div></div></div></div></div></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-22" type="checkbox" ><label for="sk-estimator-id-22" class="sk-toggleable__label sk-toggleable__label-arrow">LogisticRegression</label><div class="sk-toggleable__content"><pre>LogisticRegression(max_iter=1000, random_state=9001, solver=&#x27;liblinear&#x27;)</pre></div></div></div></div></div></div></div>




```python
# let's grab the prediction for the classification report
LR_pred = pipe_LR.predict(split_test)
```


```python
# here we see an accuracy of 62%, with a 74% accuracy for popular songs and only 30% accurate for unpopular songs
print(classification_report(Y_test, LR_pred))
```

                  precision    recall  f1-score   support
    
               0       0.59      0.20      0.30       163
               1       0.62      0.90      0.74       237
    
        accuracy                           0.62       400
       macro avg       0.61      0.55      0.52       400
    weighted avg       0.61      0.62      0.56       400
    
    

Based off the classification report, I can see a glowing problem with the lack of data points. Only have 400 available for cross validation and 1600 for model building limits the ability to build an effective model. Finding a way to scale up this project will greatly improve our ability to create a working model to identify popular songs on Spotify. 

This dataset is mildly skewed. It is built from the top 2000 songs, so there is already a lean towards popularity. Comparing it to Spotify as a whole would also greatly benefit the overall performance.

Long term, this model seems to be able to identify popular songs fairly well with only a 1/4 false positive. Connecting to the Spotify API and pulling data routinely would help flesh this model out as well.

External variables to consider are how popular the artist is before the song comes out, and if the song was used in a movie recently. These 2 variables could drastically improve the popularity score of the individual song as well. 


```python

```
