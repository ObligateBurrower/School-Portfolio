```python
# Justin Madsen
# DSC 540 / Catherine Williams

# import the boys
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import ssl

import urllib.request, urllib.parse, urllib.error

import json

import os

import requests

import regex as re

import sqlite3
```


```python
# import the csv. this has sales from 2006-2010
house_sales = pd.read_csv('house_sales.csv')
```


```python
# .head to test
house_sales.head(2)
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
      <th>Id</th>
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>Alley</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>...</th>
      <th>PoolArea</th>
      <th>PoolQC</th>
      <th>Fence</th>
      <th>MiscFeature</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>60</td>
      <td>RL</td>
      <td>65.0</td>
      <td>8450</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>208500</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>20</td>
      <td>RL</td>
      <td>80.0</td>
      <td>9600</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>5</td>
      <td>2007</td>
      <td>WD</td>
      <td>Normal</td>
      <td>181500</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 81 columns</p>
</div>




```python
# Unfortunately, because the data is already "cleaned" due it being a competition from Kaggle, the transformation
# pieces aren't really necessary here. Due to the sheer size, I'll be dropping some columns that I don't think will
# be all too necessary for the purposes of this project. tl;dr - I don't know what alot of these variables are,
# so I'll be cutting them out
```


```python
# Let's drop all the columns that aren't necessary
house_sales_df = house_sales.drop(['GarageCond', 'WoodDeckSF', 'OpenPorchSF', 
                                      'EnclosedPorch', 'MiscFeature', 'LotConfig', 
                                      'BldgType', 'MiscVal', 'GarageYrBlt', 'GarageQual',
                                     '3SsnPorch', 'ScreenPorch', 'Fence', 'MoSold', 'SaleCondition',
                                     'PoolQC', 'Neighborhood', 'Utilities', 'LotShape', 'LandContour',
                                     'MSSubClass', 'MSZoning', 'Street', 'Alley', 'PoolArea',
                                     'LandSlope', 'Condition1', 'Condition2', 'FireplaceQu', 'GarageFinish',
                                     'RoofStyle', 'Functional', 'SaleType', 'Exterior1st', 'Exterior2nd', 
                                      'KitchenQual', 'HeatingQC', 'LowQualFinSF', 'MasVnrType', 'MasVnrArea', 
                                      'ExterCond', 'BsmtExposure','BsmtCond', 'GarageType', 'BsmtFinType1',
                                      'BsmtFinType2', 'BsmtFinSF1', 'BsmtFinSF2', 'Heating', 'LotFrontage',
                                      'BsmtQual', 'BsmtUnfSF', 'Electrical', 'BsmtHalfBath', 'BsmtFullBath',
                                      'ExterQual', 'Foundation', 'RoofMatl', 'Id', 'LotArea', 'HouseStyle', 
                                      'OverallQual', 'OverallCond','YearBuilt', 'YearRemodAdd', 'TotalBsmtSF',
                                      'CentralAir', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'FullBath', 'HalfBath', 
                                      'BedroomAbvGr','KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars',
                                      'GarageArea', 'PavedDrive'] , axis = 1)
```


```python
# test to see if it worked
house_sales_df.head(2)
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
      <th>YrSold</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2008</td>
      <td>208500</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2007</td>
      <td>181500</td>
    </tr>
  </tbody>
</table>
</div>




```python
# let's look for price outliers
plt.boxplot(house_sales_df.SalePrice)
plt.show()
```


    
![png](output_6_0.png)
    



```python
house_sales_df['SalePrice'].describe()
```




    count      1460.000000
    mean     180921.195890
    std       79442.502883
    min       34900.000000
    25%      129975.000000
    50%      163000.000000
    75%      214000.000000
    max      755000.000000
    Name: SalePrice, dtype: float64




```python
# https://www.scribbr.com/statistics/outliers/
# let's determine the IQR (214000 - 129975)
q1 = 129975
q3 = 214000
iqr = q3 - q1
```


```python
# Q3 + (1.5 * IQR)
upper_fence = q3 + (1.5 * iqr)
upper_fence
```




    340037.5




```python
# Q1 – (1.5 * IQR)
lower_fence = q1 - (1.5 * iqr)
lower_fence
```




    3937.5




```python
outliers = [element for element in house_sales_df['SalePrice'] if element > upper_fence]
number_of_outliers = len(outliers)
number_of_outliers

# this tells me that there are 61 outliers. However, this could be due to being closer to the end of the dataset
# where prices would naturally be higher. I don't intend on removing this outliers, as I would like to compare them
# to the data pulled from the Zillow API later to see how prices have moved overall.
```




    61




```python
# assign an object to hold the webpage. using open,'r' will let us read the page and store it into the object
webpage = open("Economy of the United States - Wikipedia.html", "r")

# assign a variable to a BeautifulSoup of the webpage
soup = BeautifulSoup(webpage)

# close the webpage, since we no longer need it open after the soup has been made
webpage.close()
```


```python
# here we create an object to hold all tables with the class "wikitable"
data_table = soup.find("table", {"class": "wikitable"})
print(type(data_table))
```

    <class 'bs4.element.Tag'>
    


```python
# pull the wiki_df, this will turn it into a list
wiki_df=pd.read_html(str(data_table))
```


```python
# make it a nice looking df
wiki_df = pd.DataFrame(wiki_df[0])
wiki_df.head(2)
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
      <th>Year</th>
      <th>GDP (in Bil. US$PPP)</th>
      <th>GDP per capita (in US$ PPP)</th>
      <th>GDP (in Bil. US$nominal)</th>
      <th>GDP per capita (in US$ nominal)</th>
      <th>GDP growth (real)</th>
      <th>Inflation rate (in Percent)</th>
      <th>Unemployment (in Percent)</th>
      <th>Government debt (in % of GDP)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1980</td>
      <td>2857.3</td>
      <td>12552.9</td>
      <td>2857.3</td>
      <td>12552.9</td>
      <td>-0.3%</td>
      <td>13.5%</td>
      <td>7.2%</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1981</td>
      <td>3207.0</td>
      <td>13948.7</td>
      <td>3207.0</td>
      <td>13948.7</td>
      <td>2.5%</td>
      <td>10.4%</td>
      <td>7.6%</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# I only pulled this data set for the inflation rate, so I'm going to drop several columns for
# the steps of transformations
wiki_df = wiki_df.drop('GDP (in Bil. US$PPP)', axis = 1)
```


```python
# this is good information, however as stated above I'm looking for inflation rate to compare house price growth
wiki_df = wiki_df.drop('GDP per capita (in US$ PPP)', axis = 1)
```


```python
wiki_df = wiki_df.drop('GDP (in Bil. US$nominal)', axis = 1)
```


```python
wiki_df = wiki_df.drop('GDP per capita (in US$ nominal)', axis = 1)
```


```python
wiki_df = wiki_df.drop('GDP growth (real)', axis = 1)
```


```python
# this column would make an interesting comparison. How does unemployment rate affect home sales?
wiki_df = wiki_df.drop('Unemployment (in Percent)', axis = 1)
```


```python
# for some reason, I was unable to drop the gov debt directly
wiki_df.columns
```




    Index(['Year', 'Inflation rate (in Percent)', 'Government debt (in % of GDP)'], dtype='object')




```python
# in lieu of using .drop, I just assigned the 2 columns I cared about into the df and called it good.
wiki_df = wiki_df[['Year', 'Inflation rate (in Percent)']]
```


```python
# rename the column for easier reference
wiki_df.columns = ['Year', 'Inflation']
```


```python
# interesting enough, I am able to see the forecasted growth of the economy through 2027. Meanwhile, I know that
# zillow is calling for a massive increase in housing prices over the next 2 years. 
wiki_df
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
      <th>Year</th>
      <th>Inflation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1980</td>
      <td>13.5%</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1981</td>
      <td>10.4%</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1982</td>
      <td>6.2%</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1983</td>
      <td>3.2%</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1984</td>
      <td>4.4%</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1985</td>
      <td>3.5%</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1986</td>
      <td>1.9%</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1987</td>
      <td>3.6%</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1988</td>
      <td>4.1%</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1989</td>
      <td>4.8%</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1990</td>
      <td>5.4%</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1991</td>
      <td>4.2%</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1992</td>
      <td>3.0%</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1993</td>
      <td>3.0%</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1994</td>
      <td>2.6%</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1995</td>
      <td>2.8%</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1996</td>
      <td>2.9%</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1997</td>
      <td>2.3%</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1998</td>
      <td>1.5%</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1999</td>
      <td>2.2%</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2000</td>
      <td>3.4%</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2001</td>
      <td>2.8%</td>
    </tr>
    <tr>
      <th>22</th>
      <td>2002</td>
      <td>1.6%</td>
    </tr>
    <tr>
      <th>23</th>
      <td>2003</td>
      <td>2.3%</td>
    </tr>
    <tr>
      <th>24</th>
      <td>2004</td>
      <td>2.7%</td>
    </tr>
    <tr>
      <th>25</th>
      <td>2005</td>
      <td>3.4%</td>
    </tr>
    <tr>
      <th>26</th>
      <td>2006</td>
      <td>3.2%</td>
    </tr>
    <tr>
      <th>27</th>
      <td>2007</td>
      <td>2.9%</td>
    </tr>
    <tr>
      <th>28</th>
      <td>2008</td>
      <td>3.8%</td>
    </tr>
    <tr>
      <th>29</th>
      <td>2009</td>
      <td>-0.3%</td>
    </tr>
    <tr>
      <th>30</th>
      <td>2010</td>
      <td>1.6%</td>
    </tr>
    <tr>
      <th>31</th>
      <td>2011</td>
      <td>3.1%</td>
    </tr>
    <tr>
      <th>32</th>
      <td>2012</td>
      <td>2.1%</td>
    </tr>
    <tr>
      <th>33</th>
      <td>2013</td>
      <td>1.5%</td>
    </tr>
    <tr>
      <th>34</th>
      <td>2014</td>
      <td>1.6%</td>
    </tr>
    <tr>
      <th>35</th>
      <td>2015</td>
      <td>0.1%</td>
    </tr>
    <tr>
      <th>36</th>
      <td>2016</td>
      <td>1.3%</td>
    </tr>
    <tr>
      <th>37</th>
      <td>2017</td>
      <td>2.1%</td>
    </tr>
    <tr>
      <th>38</th>
      <td>2018</td>
      <td>2.4%</td>
    </tr>
    <tr>
      <th>39</th>
      <td>2019</td>
      <td>1.8%</td>
    </tr>
    <tr>
      <th>40</th>
      <td>2020</td>
      <td>1.2%</td>
    </tr>
    <tr>
      <th>41</th>
      <td>2021</td>
      <td>4.7%</td>
    </tr>
    <tr>
      <th>42</th>
      <td>2022</td>
      <td>7.7%</td>
    </tr>
    <tr>
      <th>43</th>
      <td>2023</td>
      <td>2.9%</td>
    </tr>
    <tr>
      <th>44</th>
      <td>2024</td>
      <td>2.3%</td>
    </tr>
    <tr>
      <th>45</th>
      <td>2025</td>
      <td>2.0%</td>
    </tr>
    <tr>
      <th>46</th>
      <td>2026</td>
      <td>2.0%</td>
    </tr>
    <tr>
      <th>47</th>
      <td>2027</td>
      <td>2.0%</td>
    </tr>
  </tbody>
</table>
</div>




```python
# open the consumer key file, then load the json into an object
with open('zillow.json') as f:
    key = json.load(f)
    api_key = key['apikey']
```


```python
# set base url for api
base_url = 'https://apis.estated.com/v4/property'
```


```python
# provide an address to pull
address = '360 Platteview Dr'
city = 'Springfield'
state = 'NE'
zip_code = '68059'
```


```python
# set parameters
params = (('token', api_key), ('street_address', address),
         ('city', city), ('state', state), ('zip_code', zip_code))
```


```python
# get the json response and make it legible
response = requests.get(base_url, params=params)
response_json = response.json()
```


```python
# normalize the data, this isn't very readable in it's current state
estate_df = pd.json_normalize(data=response_json['data'])
estate_df.head(2)
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
      <th>valuation</th>
      <th>taxes</th>
      <th>assessments</th>
      <th>market_assessments</th>
      <th>deeds</th>
      <th>metadata.publishing_date</th>
      <th>address.street_number</th>
      <th>address.street_pre_direction</th>
      <th>address.street_name</th>
      <th>address.street_suffix</th>
      <th>...</th>
      <th>owner.name</th>
      <th>owner.second_name</th>
      <th>owner.unit_type</th>
      <th>owner.unit_number</th>
      <th>owner.formatted_street_address</th>
      <th>owner.city</th>
      <th>owner.state</th>
      <th>owner.zip_code</th>
      <th>owner.zip_plus_four_code</th>
      <th>owner.owner_occupied</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>None</td>
      <td>[{'year': 2021, 'amount': 4827, 'exemptions': ...</td>
      <td>[{'year': 2021, 'land_value': 24000, 'improvem...</td>
      <td>[{'year': 2021, 'land_value': 24000, 'improvem...</td>
      <td>[{'document_type': 'WARRANTY DEED', 'recording...</td>
      <td>2021-09-01</td>
      <td>360</td>
      <td>None</td>
      <td>PLATTEVIEW</td>
      <td>DR</td>
      <td>...</td>
      <td>MADSEN, JUSTIN</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>360 PLATTEVIEW DR</td>
      <td>SPRINGFIELD</td>
      <td>NE</td>
      <td>68059</td>
      <td>4758</td>
      <td>YES</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 91 columns</p>
</div>




```python
print('Keys available: {}', response_json['data'].keys())
```

    Keys available: {} dict_keys(['metadata', 'address', 'parcel', 'structure', 'valuation', 'taxes', 'assessments', 'market_assessments', 'owner', 'deeds'])
    


```python
# make a dictionary of the data
deeds_dict = response_json['data']['deeds']
df_deeds = pd.DataFrame(deeds_dict)
df_deeds
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
      <th>document_type</th>
      <th>recording_date</th>
      <th>original_contract_date</th>
      <th>deed_book</th>
      <th>deed_page</th>
      <th>document_id</th>
      <th>sale_price</th>
      <th>sale_price_description</th>
      <th>transfer_tax</th>
      <th>distressed_sale</th>
      <th>...</th>
      <th>buyer_state</th>
      <th>buyer_zip_code</th>
      <th>buyer_zip_plus_four_code</th>
      <th>lender_name</th>
      <th>lender_type</th>
      <th>loan_amount</th>
      <th>loan_type</th>
      <th>loan_due_date</th>
      <th>loan_finance_type</th>
      <th>loan_interest_rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>WARRANTY DEED</td>
      <td>2020-06-02</td>
      <td>2020-05-28</td>
      <td>None</td>
      <td>None</td>
      <td>2020-14897</td>
      <td>230000</td>
      <td>SALES PRICE COMPUTED FROM COUNTY TRANSFER TAX ...</td>
      <td>517.5</td>
      <td>False</td>
      <td>...</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>MORTGAGE RESEARCH CENTER LLC</td>
      <td>MORTGAGE COMPANY</td>
      <td>234315</td>
      <td>VETERAN AFFAIRS</td>
      <td>2050-07-01</td>
      <td>None</td>
      <td>None</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 39 columns</p>
</div>




```python
# drop all the columns I don't need. I just care about year sale and sale price
df_deeds = df_deeds.drop(['document_type', 'recording_date', 'original_contract_date',
       'deed_book', 'deed_page', 'document_id',
       'sale_price_description', 'transfer_tax', 'distressed_sale',
       'real_estate_owned', 'seller_first_name', 'seller_last_name',
       'seller2_first_name', 'seller2_last_name','lender_name', 'lender_type', 'loan_amount', 'loan_type',
       'loan_due_date', 'loan_finance_type', 'loan_interest_rate', 'seller_address', 'seller_unit_number', 
                          'seller_city', 'seller_state',
       'seller_zip_code', 'seller_zip_plus_four_code', 'buyer_first_name',
       'buyer_last_name', 'buyer2_first_name', 'buyer2_last_name', 'buyer_unit_type',
       'buyer_unit_number', 'buyer_city', 'buyer_state', 'buyer_zip_code',
       'buyer_zip_plus_four_code', 'buyer_address'], axis=1)
```


```python
# add the year of the sale
df_deeds['year'] = 2020
```


```python
# reorg the columns
api_df = df_deeds[['year', 'sale_price']]
```


```python
# test to see if it worked
api_df
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
      <th>year</th>
      <th>sale_price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2020</td>
      <td>230000</td>
    </tr>
  </tbody>
</table>
</div>




```python
api_df.columns = ['YrSold', 'SalePrice']
```


```python
# set an object to hold the connection to the sql database
conn = sqlite3.connect("house_sales_db")
cursor = conn.cursor()
```


```python
# let's create our empty table
cursor.execute('CREATE TABLE IF NOT EXISTS house_sales_df (YrSold number, SalePrice number)')
conn.commit()
```


```python
# let's push the df to the db
house_sales_df.to_sql('house_sales_df', conn, if_exists='replace', index = False)
```


```python
# let's check to see if it worked
cursor.execute('''  
SELECT * FROM house_sales_df
          ''')

for row in cursor.fetchall():
    print(row)
```

    (2008, 208500)
    (2007, 181500)
    (2008, 223500)
    (2006, 140000)
    (2008, 250000)
    (2009, 143000)
    (2007, 307000)
    (2009, 200000)
    (2008, 129900)
    (2008, 118000)
    (2008, 129500)
    (2006, 345000)
    (2008, 144000)
    (2007, 279500)
    (2008, 157000)
    (2007, 132000)
    (2010, 149000)
    (2006, 90000)
    (2008, 159000)
    (2009, 139000)
    (2006, 325300)
    (2007, 139400)
    (2008, 230000)
    (2007, 129900)
    (2010, 154000)
    (2009, 256300)
    (2010, 134800)
    (2010, 306000)
    (2006, 207500)
    (2008, 68500)
    (2008, 40000)
    (2008, 149350)
    (2008, 179900)
    (2010, 165500)
    (2007, 277500)
    (2006, 309000)
    (2009, 145000)
    (2009, 153000)
    (2010, 109000)
    (2008, 82000)
    (2006, 160000)
    (2007, 170000)
    (2007, 144000)
    (2008, 130250)
    (2006, 141000)
    (2010, 319900)
    (2009, 239686)
    (2007, 249700)
    (2009, 113000)
    (2007, 127000)
    (2007, 177000)
    (2006, 114500)
    (2010, 110000)
    (2006, 385000)
    (2007, 130000)
    (2008, 180500)
    (2009, 172500)
    (2006, 196500)
    (2006, 438780)
    (2008, 124900)
    (2006, 158000)
    (2007, 101000)
    (2007, 202500)
    (2010, 140000)
    (2009, 219500)
    (2007, 317000)
    (2010, 180000)
    (2007, 226000)
    (2010, 80000)
    (2006, 225000)
    (2007, 244000)
    (2007, 129500)
    (2009, 185000)
    (2010, 144900)
    (2010, 107400)
    (2009, 91000)
    (2008, 135750)
    (2008, 127000)
    (2010, 136500)
    (2009, 110000)
    (2009, 193500)
    (2006, 153500)
    (2008, 245000)
    (2007, 126500)
    (2009, 168500)
    (2006, 260000)
    (2009, 174000)
    (2009, 164500)
    (2009, 85000)
    (2007, 123600)
    (2006, 109900)
    (2006, 98600)
    (2009, 163500)
    (2007, 133900)
    (2007, 204750)
    (2009, 185000)
    (2006, 214000)
    (2007, 94750)
    (2010, 83000)
    (2010, 128950)
    (2010, 205000)
    (2010, 178000)
    (2009, 118964)
    (2010, 198900)
    (2007, 169500)
    (2008, 250000)
    (2007, 100000)
    (2008, 115000)
    (2007, 115000)
    (2010, 190000)
    (2006, 136900)
    (2010, 180000)
    (2007, 383970)
    (2007, 217000)
    (2007, 259500)
    (2007, 176000)
    (2009, 139000)
    (2007, 155000)
    (2010, 320000)
    (2006, 163990)
    (2006, 180000)
    (2007, 100000)
    (2008, 136000)
    (2008, 153900)
    (2009, 181000)
    (2006, 84500)
    (2007, 128000)
    (2007, 87000)
    (2006, 155000)
    (2006, 150000)
    (2006, 226000)
    (2009, 244000)
    (2007, 150750)
    (2009, 220000)
    (2006, 180000)
    (2008, 174000)
    (2007, 143000)
    (2006, 171000)
    (2008, 230000)
    (2009, 231500)
    (2010, 115000)
    (2006, 260000)
    (2010, 166000)
    (2009, 204000)
    (2006, 125000)
    (2006, 130000)
    (2009, 105000)
    (2010, 222500)
    (2008, 141000)
    (2006, 115000)
    (2007, 122000)
    (2008, 372402)
    (2006, 190000)
    (2008, 235000)
    (2006, 125000)
    (2008, 79000)
    (2006, 109500)
    (2010, 269500)
    (2010, 254900)
    (2006, 320000)
    (2008, 162500)
    (2008, 412500)
    (2010, 220000)
    (2007, 103200)
    (2007, 152000)
    (2008, 127500)
    (2009, 190000)
    (2007, 325624)
    (2007, 183500)
    (2006, 228000)
    (2007, 128500)
    (2010, 215000)
    (2006, 239000)
    (2008, 163000)
    (2008, 184000)
    (2007, 243000)
    (2007, 211000)
    (2006, 172500)
    (2009, 501837)
    (2007, 100000)
    (2007, 177000)
    (2006, 200100)
    (2007, 120000)
    (2008, 200000)
    (2006, 127000)
    (2006, 475000)
    (2009, 173000)
    (2009, 135000)
    (2009, 153337)
    (2008, 286000)
    (2007, 315000)
    (2007, 184000)
    (2009, 192000)
    (2006, 130000)
    (2008, 127000)
    (2009, 148500)
    (2007, 311872)
    (2006, 235000)
    (2009, 104000)
    (2009, 274900)
    (2010, 140000)
    (2008, 171500)
    (2006, 112000)
    (2008, 149000)
    (2009, 110000)
    (2009, 180500)
    (2007, 143900)
    (2008, 141000)
    (2007, 277000)
    (2008, 145000)
    (2008, 98000)
    (2010, 186000)
    (2009, 252678)
    (2006, 156000)
    (2010, 161750)
    (2006, 134450)
    (2008, 210000)
    (2006, 107000)
    (2008, 311500)
    (2006, 167240)
    (2006, 204900)
    (2009, 200000)
    (2006, 179900)
    (2009, 97000)
    (2009, 386250)
    (2009, 112000)
    (2007, 290000)
    (2008, 106000)
    (2010, 125000)
    (2009, 192500)
    (2010, 148000)
    (2009, 403000)
    (2006, 94500)
    (2010, 128200)
    (2010, 216500)
    (2008, 89500)
    (2010, 185500)
    (2010, 194500)
    (2007, 318000)
    (2010, 113000)
    (2010, 262500)
    (2007, 110500)
    (2006, 79000)
    (2009, 120000)
    (2010, 205000)
    (2006, 241500)
    (2006, 137000)
    (2006, 140000)
    (2007, 180000)
    (2007, 277000)
    (2010, 76500)
    (2007, 235000)
    (2008, 173000)
    (2007, 158000)
    (2010, 145000)
    (2006, 230000)
    (2008, 207500)
    (2009, 220000)
    (2008, 231500)
    (2008, 97000)
    (2009, 176000)
    (2007, 276000)
    (2006, 151000)
    (2010, 130000)
    (2008, 73000)
    (2008, 175500)
    (2006, 185000)
    (2008, 179500)
    (2008, 120500)
    (2007, 148000)
    (2006, 266000)
    (2008, 241500)
    (2010, 290000)
    (2009, 139000)
    (2007, 124500)
    (2009, 205000)
    (2010, 201000)
    (2010, 141000)
    (2007, 415298)
    (2008, 192000)
    (2007, 228500)
    (2006, 185000)
    (2009, 207500)
    (2009, 244600)
    (2007, 179200)
    (2007, 164700)
    (2006, 159000)
    (2006, 88000)
    (2010, 122000)
    (2007, 153575)
    (2006, 233230)
    (2008, 135900)
    (2009, 131000)
    (2006, 235000)
    (2009, 167000)
    (2006, 142500)
    (2007, 152000)
    (2007, 239000)
    (2007, 175000)
    (2009, 158500)
    (2006, 157000)
    (2007, 267000)
    (2006, 205000)
    (2006, 149900)
    (2008, 295000)
    (2007, 305900)
    (2007, 225000)
    (2008, 89500)
    (2009, 82500)
    (2006, 360000)
    (2006, 165600)
    (2009, 132000)
    (2006, 119900)
    (2009, 375000)
    (2006, 178000)
    (2009, 188500)
    (2009, 260000)
    (2007, 270000)
    (2009, 260000)
    (2009, 187500)
    (2006, 342643)
    (2007, 354000)
    (2007, 301000)
    (2006, 126175)
    (2010, 242000)
    (2007, 87000)
    (2008, 324000)
    (2006, 145250)
    (2009, 214500)
    (2009, 78000)
    (2007, 119000)
    (2007, 139000)
    (2009, 284000)
    (2008, 207000)
    (2008, 192000)
    (2008, 228950)
    (2007, 377426)
    (2008, 214000)
    (2006, 202500)
    (2009, 155000)
    (2010, 202900)
    (2009, 82000)
    (2006, 87500)
    (2008, 266000)
    (2010, 85000)
    (2006, 140200)
    (2007, 151500)
    (2009, 157500)
    (2008, 154000)
    (2006, 437154)
    (2007, 318061)
    (2006, 190000)
    (2008, 95000)
    (2010, 105900)
    (2006, 140000)
    (2007, 177500)
    (2009, 173000)
    (2007, 134000)
    (2006, 130000)
    (2006, 280000)
    (2007, 156000)
    (2008, 145000)
    (2009, 198500)
    (2009, 118000)
    (2006, 190000)
    (2009, 147000)
    (2009, 159000)
    (2008, 165000)
    (2010, 132000)
    (2010, 162000)
    (2006, 172400)
    (2008, 134432)
    (2010, 125000)
    (2009, 123000)
    (2007, 219500)
    (2009, 61000)
    (2006, 148000)
    (2007, 340000)
    (2010, 394432)
    (2009, 179000)
    (2010, 127000)
    (2006, 187750)
    (2007, 213500)
    (2009, 76000)
    (2007, 240000)
    (2010, 192000)
    (2006, 81000)
    (2009, 125000)
    (2008, 191000)
    (2008, 426000)
    (2008, 119000)
    (2009, 215000)
    (2007, 106500)
    (2006, 100000)
    (2007, 109000)
    (2010, 129000)
    (2009, 123000)
    (2007, 169500)
    (2007, 67000)
    (2009, 241000)
    (2008, 245500)
    (2006, 164990)
    (2008, 108000)
    (2006, 258000)
    (2007, 168000)
    (2009, 150000)
    (2008, 115000)
    (2008, 177000)
    (2007, 280000)
    (2008, 339750)
    (2009, 60000)
    (2006, 145000)
    (2010, 222000)
    (2010, 115000)
    (2008, 228000)
    (2007, 181134)
    (2006, 149500)
    (2007, 239000)
    (2007, 126000)
    (2010, 142000)
    (2008, 206300)
    (2009, 215000)
    (2008, 113000)
    (2008, 315000)
    (2008, 139000)
    (2009, 135000)
    (2009, 275000)
    (2008, 109008)
    (2007, 195400)
    (2009, 175000)
    (2008, 85400)
    (2008, 79900)
    (2007, 122500)
    (2008, 181000)
    (2008, 81000)
    (2009, 212000)
    (2006, 116000)
    (2009, 119000)
    (2007, 90350)
    (2009, 110000)
    (2009, 555000)
    (2008, 118000)
    (2008, 162900)
    (2007, 172500)
    (2008, 210000)
    (2009, 127500)
    (2010, 190000)
    (2006, 199900)
    (2006, 119500)
    (2007, 120000)
    (2006, 110000)
    (2006, 280000)
    (2007, 204000)
    (2009, 210000)
    (2006, 188000)
    (2007, 175500)
    (2008, 98000)
    (2008, 256000)
    (2008, 161000)
    (2009, 110000)
    (2009, 263435)
    (2009, 155000)
    (2009, 62383)
    (2008, 188700)
    (2009, 124000)
    (2006, 178740)
    (2007, 167000)
    (2007, 146500)
    (2007, 250000)
    (2008, 187000)
    (2010, 212000)
    (2007, 190000)
    (2008, 148000)
    (2007, 440000)
    (2007, 251000)
    (2007, 132500)
    (2008, 208900)
    (2007, 380000)
    (2009, 297000)
    (2007, 89471)
    (2006, 326000)
    (2006, 374000)
    (2009, 155000)
    (2006, 164000)
    (2007, 132500)
    (2009, 147000)
    (2007, 156000)
    (2007, 175000)
    (2006, 160000)
    (2009, 86000)
    (2008, 115000)
    (2006, 133000)
    (2006, 172785)
    (2008, 155000)
    (2009, 91300)
    (2009, 34900)
    (2007, 430000)
    (2008, 184000)
    (2009, 130000)
    (2007, 120000)
    (2007, 113000)
    (2008, 226700)
    (2007, 140000)
    (2010, 289000)
    (2009, 147000)
    (2009, 124500)
    (2006, 215000)
    (2009, 208300)
    (2008, 161000)
    (2009, 124500)
    (2009, 164900)
    (2006, 202665)
    (2006, 129900)
    (2007, 134000)
    (2007, 96500)
    (2009, 402861)
    (2009, 158000)
    (2009, 265000)
    (2007, 211000)
    (2009, 234000)
    (2008, 106250)
    (2007, 150000)
    (2006, 159000)
    (2007, 184750)
    (2007, 315750)
    (2006, 176000)
    (2007, 132000)
    (2008, 446261)
    (2007, 86000)
    (2007, 200624)
    (2008, 175000)
    (2008, 128000)
    (2010, 107500)
    (2007, 39300)
    (2006, 178000)
    (2008, 107500)
    (2008, 188000)
    (2008, 111250)
    (2006, 158000)
    (2010, 272000)
    (2009, 315000)
    (2007, 248000)
    (2009, 213250)
    (2007, 133000)
    (2006, 179665)
    (2006, 229000)
    (2007, 210000)
    (2007, 129500)
    (2008, 125000)
    (2009, 263000)
    (2008, 140000)
    (2008, 112500)
    (2009, 255500)
    (2009, 108000)
    (2008, 284000)
    (2006, 113000)
    (2006, 141000)
    (2006, 108000)
    (2008, 175000)
    (2006, 234000)
    (2010, 121500)
    (2006, 170000)
    (2008, 108000)
    (2008, 185000)
    (2006, 268000)
    (2010, 128000)
    (2008, 325000)
    (2010, 214000)
    (2009, 316600)
    (2006, 135960)
    (2008, 142600)
    (2006, 120000)
    (2009, 224500)
    (2007, 170000)
    (2007, 139000)
    (2008, 118500)
    (2009, 145000)
    (2006, 164500)
    (2008, 146000)
    (2008, 131500)
    (2007, 181900)
    (2009, 253293)
    (2007, 118500)
    (2008, 325000)
    (2009, 133000)
    (2006, 369900)
    (2008, 130000)
    (2009, 137000)
    (2009, 143000)
    (2008, 79500)
    (2008, 185900)
    (2009, 451950)
    (2008, 138000)
    (2009, 140000)
    (2008, 110000)
    (2006, 319000)
    (2006, 114504)
    (2007, 194201)
    (2006, 217500)
    (2008, 151000)
    (2006, 275000)
    (2007, 141000)
    (2006, 220000)
    (2010, 151000)
    (2008, 221000)
    (2009, 205000)
    (2009, 152000)
    (2006, 225000)
    (2007, 359100)
    (2007, 118500)
    (2009, 313000)
    (2007, 148000)
    (2009, 261500)
    (2007, 147000)
    (2010, 75500)
    (2010, 137500)
    (2006, 183200)
    (2008, 105500)
    (2007, 314813)
    (2008, 305000)
    (2008, 67000)
    (2008, 240000)
    (2009, 135000)
    (2007, 168500)
    (2006, 165150)
    (2010, 160000)
    (2007, 139900)
    (2010, 153000)
    (2007, 135000)
    (2008, 168500)
    (2006, 124000)
    (2007, 209500)
    (2009, 82500)
    (2007, 139400)
    (2010, 144000)
    (2007, 200000)
    (2009, 60000)
    (2009, 93000)
    (2008, 85000)
    (2006, 264561)
    (2008, 274000)
    (2007, 226000)
    (2009, 345000)
    (2007, 152000)
    (2009, 370878)
    (2007, 143250)
    (2008, 98300)
    (2008, 155000)
    (2010, 155000)
    (2007, 84500)
    (2008, 205950)
    (2009, 108000)
    (2009, 191000)
    (2008, 135000)
    (2008, 350000)
    (2010, 88000)
    (2008, 145500)
    (2008, 149000)
    (2010, 97500)
    (2009, 167000)
    (2007, 197900)
    (2009, 402000)
    (2009, 110000)
    (2008, 137500)
    (2006, 423000)
    (2006, 230500)
    (2007, 129000)
    (2008, 193500)
    (2006, 168000)
    (2006, 137500)
    (2009, 173500)
    (2009, 103600)
    (2006, 165000)
    (2007, 257500)
    (2008, 140000)
    (2009, 148500)
    (2006, 87000)
    (2009, 109500)
    (2009, 372500)
    (2007, 128500)
    (2010, 143000)
    (2009, 159434)
    (2008, 173000)
    (2007, 285000)
    (2010, 221000)
    (2007, 207500)
    (2007, 227875)
    (2007, 148800)
    (2007, 392000)
    (2007, 194700)
    (2008, 141000)
    (2007, 755000)
    (2006, 335000)
    (2006, 108480)
    (2009, 141500)
    (2006, 176000)
    (2006, 89000)
    (2006, 123500)
    (2010, 138500)
    (2008, 196000)
    (2006, 312500)
    (2006, 140000)
    (2006, 361919)
    (2010, 140000)
    (2010, 213000)
    (2010, 55000)
    (2007, 302000)
    (2009, 254000)
    (2007, 179540)
    (2008, 109900)
    (2008, 52000)
    (2010, 102776)
    (2008, 189000)
    (2006, 129000)
    (2010, 130500)
    (2009, 165000)
    (2007, 159500)
    (2008, 157000)
    (2008, 341000)
    (2006, 128500)
    (2006, 275000)
    (2010, 143000)
    (2009, 124500)
    (2008, 135000)
    (2009, 320000)
    (2009, 120500)
    (2009, 222000)
    (2009, 194500)
    (2009, 110000)
    (2009, 103000)
    (2010, 236500)
    (2007, 187500)
    (2008, 222500)
    (2009, 131400)
    (2007, 108000)
    (2006, 163000)
    (2006, 93500)
    (2006, 239900)
    (2009, 179000)
    (2009, 190000)
    (2007, 132000)
    (2008, 142000)
    (2007, 179000)
    (2009, 175000)
    (2008, 180000)
    (2008, 299800)
    (2009, 236000)
    (2009, 265979)
    (2010, 260400)
    (2009, 98000)
    (2010, 96500)
    (2007, 162000)
    (2006, 217000)
    (2006, 275500)
    (2009, 156000)
    (2009, 172500)
    (2009, 212000)
    (2010, 158900)
    (2008, 179400)
    (2007, 290000)
    (2009, 127500)
    (2009, 100000)
    (2010, 215200)
    (2009, 337000)
    (2006, 270000)
    (2008, 264132)
    (2010, 196500)
    (2008, 160000)
    (2006, 216837)
    (2010, 538000)
    (2009, 134900)
    (2006, 102000)
    (2010, 107000)
    (2007, 114500)
    (2007, 395000)
    (2009, 162000)
    (2006, 221500)
    (2006, 142500)
    (2007, 144000)
    (2006, 135000)
    (2007, 176000)
    (2006, 175900)
    (2009, 187100)
    (2009, 165500)
    (2008, 128000)
    (2009, 161500)
    (2010, 139000)
    (2010, 233000)
    (2008, 107900)
    (2007, 187500)
    (2009, 160200)
    (2007, 146800)
    (2007, 269790)
    (2007, 225000)
    (2008, 194500)
    (2010, 171000)
    (2007, 143500)
    (2008, 110000)
    (2009, 485000)
    (2007, 175000)
    (2008, 200000)
    (2007, 109900)
    (2008, 189000)
    (2009, 582933)
    (2006, 118000)
    (2008, 227680)
    (2006, 135500)
    (2009, 223500)
    (2006, 159950)
    (2009, 106000)
    (2006, 181000)
    (2008, 144500)
    (2010, 55993)
    (2007, 157900)
    (2006, 116000)
    (2010, 224900)
    (2006, 137000)
    (2008, 271000)
    (2010, 155000)
    (2010, 224000)
    (2008, 183000)
    (2009, 93000)
    (2007, 225000)
    (2009, 139500)
    (2006, 232600)
    (2008, 385000)
    (2008, 109500)
    (2009, 189000)
    (2009, 185000)
    (2006, 147400)
    (2008, 166000)
    (2006, 151000)
    (2010, 237000)
    (2009, 167000)
    (2008, 139950)
    (2010, 128000)
    (2007, 153500)
    (2008, 100000)
    (2008, 144000)
    (2008, 130500)
    (2008, 140000)
    (2008, 157500)
    (2008, 174900)
    (2007, 141000)
    (2008, 153900)
    (2007, 171000)
    (2009, 213000)
    (2009, 133500)
    (2008, 240000)
    (2007, 187000)
    (2007, 131500)
    (2006, 215000)
    (2007, 164000)
    (2009, 158000)
    (2006, 170000)
    (2010, 127000)
    (2008, 147000)
    (2009, 174000)
    (2009, 152000)
    (2006, 250000)
    (2007, 189950)
    (2010, 131500)
    (2010, 152000)
    (2009, 132500)
    (2008, 250580)
    (2009, 148500)
    (2007, 248900)
    (2007, 129000)
    (2006, 169000)
    (2010, 236000)
    (2009, 109500)
    (2010, 200500)
    (2008, 116000)
    (2009, 133000)
    (2009, 66500)
    (2007, 303477)
    (2007, 132250)
    (2009, 350000)
    (2010, 148000)
    (2009, 136500)
    (2007, 157000)
    (2007, 187500)
    (2009, 178000)
    (2006, 118500)
    (2009, 100000)
    (2008, 328900)
    (2006, 145000)
    (2008, 135500)
    (2007, 268000)
    (2009, 149500)
    (2007, 122900)
    (2009, 172500)
    (2006, 154500)
    (2008, 165000)
    (2009, 118858)
    (2008, 140000)
    (2006, 106500)
    (2009, 142953)
    (2010, 611657)
    (2006, 135000)
    (2007, 110000)
    (2009, 153000)
    (2006, 180000)
    (2006, 240000)
    (2007, 125500)
    (2010, 128000)
    (2007, 255000)
    (2006, 250000)
    (2006, 131000)
    (2009, 174000)
    (2010, 154300)
    (2009, 143500)
    (2006, 88000)
    (2007, 145000)
    (2009, 173733)
    (2007, 75000)
    (2006, 35311)
    (2009, 135000)
    (2007, 238000)
    (2008, 176500)
    (2007, 201000)
    (2008, 145900)
    (2006, 169990)
    (2008, 193000)
    (2006, 207500)
    (2008, 175000)
    (2007, 285000)
    (2008, 176000)
    (2009, 236500)
    (2006, 222000)
    (2009, 201000)
    (2009, 117500)
    (2007, 320000)
    (2009, 190000)
    (2008, 242000)
    (2006, 79900)
    (2009, 184900)
    (2009, 253000)
    (2006, 239799)
    (2010, 244400)
    (2006, 150900)
    (2009, 214000)
    (2007, 150000)
    (2007, 143000)
    (2009, 137500)
    (2009, 124900)
    (2006, 143000)
    (2007, 270000)
    (2006, 192500)
    (2010, 197500)
    (2007, 129000)
    (2006, 119900)
    (2009, 133900)
    (2008, 172000)
    (2006, 127500)
    (2007, 145000)
    (2009, 124000)
    (2007, 132000)
    (2007, 185000)
    (2010, 155000)
    (2010, 116500)
    (2008, 272000)
    (2007, 155000)
    (2009, 239000)
    (2010, 214900)
    (2007, 178900)
    (2009, 160000)
    (2008, 135000)
    (2009, 37900)
    (2006, 140000)
    (2006, 135000)
    (2009, 173000)
    (2010, 99500)
    (2008, 182000)
    (2009, 167500)
    (2006, 165000)
    (2006, 85500)
    (2007, 199900)
    (2007, 110000)
    (2009, 139000)
    (2008, 178400)
    (2009, 336000)
    (2008, 159895)
    (2008, 255900)
    (2009, 126000)
    (2008, 125000)
    (2006, 117000)
    (2010, 395192)
    (2007, 195000)
    (2006, 197000)
    (2006, 348000)
    (2009, 168000)
    (2007, 187000)
    (2006, 173900)
    (2009, 337500)
    (2006, 121600)
    (2006, 136500)
    (2009, 185000)
    (2006, 91000)
    (2010, 206000)
    (2009, 82000)
    (2007, 86000)
    (2008, 232000)
    (2007, 136905)
    (2009, 181000)
    (2008, 149900)
    (2007, 163500)
    (2009, 88000)
    (2009, 240000)
    (2006, 102000)
    (2008, 135000)
    (2010, 100000)
    (2007, 165000)
    (2009, 85000)
    (2007, 119200)
    (2009, 227000)
    (2009, 203000)
    (2009, 187500)
    (2007, 160000)
    (2006, 213490)
    (2008, 176000)
    (2006, 194000)
    (2007, 87000)
    (2008, 191000)
    (2008, 287000)
    (2007, 112500)
    (2010, 167500)
    (2008, 293077)
    (2007, 105000)
    (2006, 118000)
    (2006, 160000)
    (2009, 197000)
    (2006, 310000)
    (2006, 230000)
    (2007, 119750)
    (2009, 84000)
    (2009, 315500)
    (2008, 287000)
    (2009, 97000)
    (2009, 80000)
    (2006, 155000)
    (2008, 173000)
    (2009, 196000)
    (2008, 262280)
    (2009, 278000)
    (2009, 139600)
    (2006, 556581)
    (2008, 145000)
    (2009, 115000)
    (2010, 84900)
    (2007, 176485)
    (2007, 200141)
    (2007, 165000)
    (2010, 144500)
    (2006, 255000)
    (2008, 180000)
    (2006, 185850)
    (2009, 248000)
    (2009, 335000)
    (2007, 220000)
    (2010, 213500)
    (2008, 81000)
    (2007, 90000)
    (2006, 110500)
    (2009, 154000)
    (2010, 328000)
    (2009, 178000)
    (2008, 167900)
    (2006, 151400)
    (2007, 135000)
    (2007, 135000)
    (2009, 154000)
    (2006, 91500)
    (2009, 159500)
    (2007, 194000)
    (2007, 219500)
    (2006, 170000)
    (2006, 138800)
    (2006, 155900)
    (2007, 126000)
    (2008, 145000)
    (2010, 133000)
    (2007, 192000)
    (2006, 160000)
    (2006, 187500)
    (2010, 147000)
    (2010, 83500)
    (2009, 252000)
    (2006, 137500)
    (2006, 197000)
    (2009, 92900)
    (2008, 160000)
    (2008, 136500)
    (2006, 146000)
    (2010, 129000)
    (2007, 176432)
    (2007, 127000)
    (2007, 170000)
    (2009, 128000)
    (2009, 157000)
    (2009, 60000)
    (2007, 119500)
    (2007, 135000)
    (2006, 159500)
    (2007, 106000)
    (2010, 325000)
    (2007, 179900)
    (2006, 274725)
    (2007, 181000)
    (2009, 280000)
    (2008, 188000)
    (2008, 205000)
    (2006, 129900)
    (2007, 134500)
    (2006, 117000)
    (2007, 318000)
    (2009, 184100)
    (2008, 130000)
    (2008, 140000)
    (2006, 133700)
    (2007, 118400)
    (2006, 212900)
    (2009, 112000)
    (2009, 118000)
    (2007, 163900)
    (2009, 115000)
    (2009, 174000)
    (2007, 259000)
    (2007, 215000)
    (2007, 140000)
    (2009, 135000)
    (2007, 93500)
    (2007, 117500)
    (2009, 239500)
    (2007, 169000)
    (2007, 102000)
    (2008, 119000)
    (2010, 94000)
    (2009, 196000)
    (2007, 144000)
    (2008, 139000)
    (2009, 197500)
    (2007, 424870)
    (2008, 80000)
    (2010, 80000)
    (2006, 149000)
    (2006, 180000)
    (2009, 174500)
    (2008, 116900)
    (2009, 143000)
    (2007, 124000)
    (2006, 149900)
    (2006, 230000)
    (2008, 120500)
    (2008, 201800)
    (2007, 218000)
    (2008, 179900)
    (2009, 230000)
    (2008, 235128)
    (2008, 185000)
    (2010, 146000)
    (2008, 224000)
    (2007, 129000)
    (2008, 108959)
    (2007, 194000)
    (2009, 233170)
    (2010, 245350)
    (2006, 173000)
    (2008, 235000)
    (2006, 625000)
    (2008, 171000)
    (2008, 163000)
    (2008, 171900)
    (2007, 200500)
    (2006, 239000)
    (2007, 285000)
    (2008, 119500)
    (2009, 115000)
    (2009, 154900)
    (2006, 93000)
    (2006, 250000)
    (2008, 392500)
    (2007, 745000)
    (2006, 120000)
    (2007, 186700)
    (2006, 104900)
    (2009, 95000)
    (2006, 262000)
    (2009, 195000)
    (2010, 189000)
    (2007, 168000)
    (2007, 174000)
    (2007, 125000)
    (2009, 165000)
    (2010, 158000)
    (2008, 176000)
    (2006, 219210)
    (2006, 144000)
    (2009, 178000)
    (2006, 148000)
    (2006, 116050)
    (2009, 197900)
    (2009, 117000)
    (2009, 213000)
    (2006, 153500)
    (2009, 271900)
    (2006, 107000)
    (2006, 200000)
    (2008, 140000)
    (2006, 290000)
    (2010, 189000)
    (2010, 164000)
    (2009, 113000)
    (2006, 145000)
    (2006, 134500)
    (2007, 125000)
    (2010, 112000)
    (2009, 229456)
    (2006, 80500)
    (2006, 91500)
    (2006, 115000)
    (2008, 134000)
    (2007, 143000)
    (2006, 137900)
    (2008, 184000)
    (2007, 145000)
    (2008, 214000)
    (2008, 147000)
    (2008, 367294)
    (2008, 127000)
    (2007, 190000)
    (2006, 132500)
    (2007, 101800)
    (2010, 142000)
    (2008, 130000)
    (2006, 138887)
    (2010, 175500)
    (2006, 195000)
    (2006, 142500)
    (2007, 265900)
    (2008, 224900)
    (2007, 248328)
    (2010, 170000)
    (2006, 465000)
    (2006, 230000)
    (2007, 178000)
    (2006, 186500)
    (2010, 169900)
    (2008, 129500)
    (2007, 119000)
    (2010, 244000)
    (2006, 171750)
    (2009, 130000)
    (2007, 294000)
    (2008, 165400)
    (2007, 127500)
    (2008, 301500)
    (2009, 99900)
    (2008, 190000)
    (2008, 151000)
    (2009, 181000)
    (2009, 128900)
    (2009, 161500)
    (2007, 180500)
    (2008, 181000)
    (2006, 183900)
    (2007, 122000)
    (2010, 378500)
    (2008, 381000)
    (2007, 144000)
    (2010, 260000)
    (2009, 185750)
    (2006, 137000)
    (2008, 177000)
    (2007, 139000)
    (2007, 137000)
    (2009, 162000)
    (2009, 197900)
    (2008, 237000)
    (2010, 68400)
    (2009, 227000)
    (2006, 180000)
    (2009, 150500)
    (2010, 139000)
    (2010, 169000)
    (2009, 132500)
    (2010, 143000)
    (2006, 190000)
    (2009, 278000)
    (2006, 281000)
    (2010, 180500)
    (2009, 119500)
    (2009, 107500)
    (2006, 162900)
    (2006, 115000)
    (2006, 138500)
    (2008, 155000)
    (2006, 140000)
    (2008, 160000)
    (2010, 154000)
    (2009, 225000)
    (2009, 177500)
    (2006, 290000)
    (2006, 232000)
    (2006, 130000)
    (2009, 325000)
    (2006, 202500)
    (2009, 138000)
    (2008, 147000)
    (2008, 179200)
    (2010, 335000)
    (2007, 203000)
    (2007, 302000)
    (2010, 333168)
    (2007, 119000)
    (2008, 206900)
    (2009, 295493)
    (2007, 208900)
    (2006, 275000)
    (2007, 111000)
    (2009, 156500)
    (2008, 72500)
    (2010, 190000)
    (2009, 82500)
    (2007, 147000)
    (2008, 55000)
    (2007, 79000)
    (2008, 130500)
    (2008, 256000)
    (2006, 176500)
    (2007, 227000)
    (2006, 132500)
    (2009, 100000)
    (2006, 125500)
    (2009, 125000)
    (2009, 167900)
    (2008, 135000)
    (2006, 52500)
    (2006, 200000)
    (2006, 128500)
    (2007, 123000)
    (2008, 155000)
    (2007, 228500)
    (2009, 177000)
    (2007, 155835)
    (2007, 108500)
    (2006, 262500)
    (2007, 283463)
    (2007, 215000)
    (2008, 122000)
    (2009, 200000)
    (2008, 171000)
    (2009, 134900)
    (2010, 410000)
    (2008, 235000)
    (2006, 170000)
    (2008, 110000)
    (2010, 149900)
    (2010, 177500)
    (2006, 315000)
    (2008, 189000)
    (2009, 260000)
    (2009, 104900)
    (2007, 156932)
    (2006, 144152)
    (2010, 216000)
    (2008, 193000)
    (2006, 127000)
    (2009, 144000)
    (2010, 232000)
    (2009, 105000)
    (2008, 165500)
    (2006, 274300)
    (2007, 466500)
    (2008, 250000)
    (2007, 239000)
    (2008, 91000)
    (2009, 117000)
    (2006, 83000)
    (2008, 167500)
    (2010, 58500)
    (2008, 237500)
    (2006, 157000)
    (2007, 112000)
    (2009, 105000)
    (2010, 125500)
    (2006, 250000)
    (2007, 136000)
    (2009, 377500)
    (2007, 131000)
    (2006, 235000)
    (2009, 124000)
    (2006, 123000)
    (2008, 163000)
    (2006, 246578)
    (2007, 281213)
    (2010, 160000)
    (2007, 137500)
    (2009, 138000)
    (2009, 137450)
    (2008, 120000)
    (2008, 193000)
    (2006, 193879)
    (2007, 282922)
    (2006, 105000)
    (2008, 275000)
    (2009, 133000)
    (2009, 112000)
    (2010, 125500)
    (2008, 215000)
    (2009, 230000)
    (2009, 140000)
    (2009, 90000)
    (2009, 257000)
    (2008, 207000)
    (2009, 175900)
    (2010, 122500)
    (2009, 340000)
    (2008, 124000)
    (2006, 223000)
    (2006, 179900)
    (2010, 127500)
    (2008, 136500)
    (2006, 274970)
    (2007, 144000)
    (2008, 142000)
    (2008, 271000)
    (2008, 140000)
    (2010, 119000)
    (2007, 182900)
    (2006, 192140)
    (2009, 143750)
    (2007, 64500)
    (2008, 186500)
    (2006, 160000)
    (2008, 174000)
    (2007, 120500)
    (2008, 394617)
    (2010, 149700)
    (2007, 197000)
    (2008, 191000)
    (2008, 149300)
    (2009, 310000)
    (2009, 121000)
    (2007, 179600)
    (2007, 129000)
    (2010, 157900)
    (2007, 240000)
    (2007, 112000)
    (2006, 92000)
    (2009, 136000)
    (2009, 287090)
    (2006, 145000)
    (2006, 84500)
    (2009, 185000)
    (2007, 175000)
    (2010, 210000)
    (2010, 266500)
    (2010, 142125)
    (2008, 147500)
    


```python
cursor.execute('CREATE TABLE IF NOT EXISTS house_sales_api (YrSold number, SalePrice number)')
conn.commit()
```


```python
# let's push the df to the db
api_df.to_sql('house_sales_api', conn, if_exists='replace', index = False)
```


```python
# let's check to see if it worked
cursor.execute('''  
SELECT * FROM house_sales_api
          ''')

for row in cursor.fetchall():
    print(row)
```

    (2020, 230000)
    


```python
cursor.execute('CREATE TABLE IF NOT EXISTS interest_rates (Year number, Inflation text)')
conn.commit()
```


```python
# let's push the df to the db
wiki_df.to_sql('interest_rates', conn, if_exists='replace', index = False)
```


```python
# let's check to see if it worked
cursor.execute('''  
SELECT * FROM interest_rates
          ''')

for row in cursor.fetchall():
    print(row)
```

    (1980, '13.5%')
    (1981, '10.4%')
    (1982, '6.2%')
    (1983, '3.2%')
    (1984, '4.4%')
    (1985, '3.5%')
    (1986, '1.9%')
    (1987, '3.6%')
    (1988, '4.1%')
    (1989, '4.8%')
    (1990, '5.4%')
    (1991, '4.2%')
    (1992, '3.0%')
    (1993, '3.0%')
    (1994, '2.6%')
    (1995, '2.8%')
    (1996, '2.9%')
    (1997, '2.3%')
    (1998, '1.5%')
    (1999, '2.2%')
    (2000, '3.4%')
    (2001, '2.8%')
    (2002, '1.6%')
    (2003, '2.3%')
    (2004, '2.7%')
    (2005, '3.4%')
    (2006, '3.2%')
    (2007, '2.9%')
    (2008, '3.8%')
    (2009, '-0.3%')
    (2010, '1.6%')
    (2011, '3.1%')
    (2012, '2.1%')
    (2013, '1.5%')
    (2014, '1.6%')
    (2015, '0.1%')
    (2016, '1.3%')
    (2017, '2.1%')
    (2018, '2.4%')
    (2019, '1.8%')
    (2020, '1.2%')
    (2021, '4.7%')
    (2022, '7.7%')
    (2023, '2.9%')
    (2024, '2.3%')
    (2025, '2.0%')
    (2026, '2.0%')
    (2027, '2.0%')
    


```python
# let's see how many houses were sold in various years
for houses, year in cursor.execute("SELECT count(*), YrSold FROM house_sales_df GROUP BY YrSold"):
    print("{} houses were sold in {}".format(houses, year))
```

    314 houses were sold in 2006
    329 houses were sold in 2007
    304 houses were sold in 2008
    338 houses were sold in 2009
    175 houses were sold in 2010
    


```python
# this will count the number of houses that sold over 200K

# I also used 2 sets of quotations for multi-line strings to improve readability
price_200k = cursor.execute("SELECT count(*) FROM house_sales_df WHERE SalePrice >200000")
# print the number of people that were counted
for house in price_200k:
    print("{} houses have sold for more than 200K".format(house[0]))
```

    427 houses have sold for more than 200K
    


```python
# let's merge the api into the df
cursor.execute('''INSERT INTO house_sales_df
SELECT * FROM house_sales_api''')
```




    <sqlite3.Cursor at 0x7fd6584f8b90>




```python
# to test if this worked, let's rerun the previous visualizations. We should see an extra print line, and
# one increase to the number over 200K.
for houses, year in cursor.execute("SELECT count(*), YrSold FROM house_sales_df GROUP BY YrSold"):
    print("{} houses were sold in {}".format(houses, year))
```

    314 houses were sold in 2006
    329 houses were sold in 2007
    304 houses were sold in 2008
    338 houses were sold in 2009
    175 houses were sold in 2010
    1 houses were sold in 2020
    


```python
# this will count the number of houses that sold over 200K

# I also used 2 sets of quotations for multi-line strings to improve readability
price_200k = cursor.execute("SELECT count(*) FROM house_sales_df WHERE SalePrice >200000")
# print the number of people that were counted
for house in price_200k:
    print("{} houses have sold for more than 200K".format(house[0]))
```

    428 houses have sold for more than 200K
    


```python
# Let's see how many houses sold for less than 120K
price_120k = cursor.execute("SELECT count(*) FROM house_sales_df WHERE SalePrice <120000")
# print the number of people that were counted
for house in price_120k:
    print("{} houses have sold for less than 120K".format(house[0]))
```

    262 houses have sold for less than 120K
    

Summary

All things considered, I struggled the most with the API files. Between Zillow's functionality dropping off and Estated not letting me request multiple locations at a time, I chose to run with just the one address I used in a previous milestone. My biggest hangup was misunderstanding the assignment's end goal as well. I was anticipating having to model the data and prepare a presentation, whereas the real goal was to learn how to prepare the data and make it useable. 

Looking at it after the fact, I can already tell that only having 4 years of sales data is restrictive for what I was originally hoping to do anyway. Ethically, house sales are a matter of public record. However, many people see personal finance as a taboo topic and don't want to discuss their finances. Especially when looking at big dips/losses during the 08-09 market crashes.
