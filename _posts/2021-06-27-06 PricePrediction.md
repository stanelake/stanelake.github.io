# House Price Prediction:
## Part 1: Data Exploration

I completed the WQU Machine Learning course 3 months ago and wanted to explore some new challenges. As a result I am exploring this Kaggle competition for leisure and am following a website cited in the references.

## Objective:
Predict house prices

## Import Python Packages:



```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import sklearn as sk
```

## Import & Clean Data:
- Two data sets are provided one for testing and the other for training.
- We import each of the csv files into a pandas dataframe and remove any unwanted details


```python
df_test = pd.read_csv('test.csv')
df_train = pd.read_csv('train.csv')
df_train.head()
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
    <tr>
      <th>2</th>
      <td>3</td>
      <td>60</td>
      <td>RL</td>
      <td>68.0</td>
      <td>11250</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>9</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>223500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>70</td>
      <td>RL</td>
      <td>60.0</td>
      <td>9550</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2006</td>
      <td>WD</td>
      <td>Abnorml</td>
      <td>140000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>60</td>
      <td>RL</td>
      <td>84.0</td>
      <td>14260</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>12</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>250000</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 81 columns</p>
</div>



### Visualise the data

- Apparently, the data has so many NaN data it may be wise not to drop them.
- Below we use a simple function to determine all the unique entries in the columns with many NaN values


```python
def unique_tally(data):
    res = None
    for i in range(len(data)):
        if not isinstance(res,dict):
            res = {}
            res[data[0]] = 1
        if data[i] in res:
            res[data[i]] +=1
        else:
            res[data[i]] = 1
    return res
```


```python
tallies = []
c = list(df_train.columns)
for col in c:
    tallies.append(unique_tally(df_train[col]))

plt.figure(figsize= (20,50))
for i in range(1,len(tallies)):
    plt.subplot(16,5,i)
    plt.bar(range(len(tallies[i])), list(tallies[i].values()), align='center', label = c[i])
    plt.xticks(range(len(tallies[i])), list(tallies[i].keys()), rotation=50)
    plt.legend()
```


    
![png](06_output_6_0.png)
    


### Replacing NaN Values:

- Clearly, not all 'NaN' values need to be discarded. 
- We will replace 'NaN' values depending on some conditions as follows:
    - If the data in the column is numerical replace NaN with zero
    - If the data in the column is of string type, replace NaN with ' '
- I will update the document once I learn how to do the above. For now letus proceed


```python
df_train=df_train.replace(np.nan,'', regex=True)
```

### Data Correlation:

- It is important to determine any interdependencies if they exist. 


```python
plt.figure(figsize=(20,20))
sb.heatmap(df_train.corr())
plt.show()
```


    
![png](06_output_10_0.png)
    



```python
plt.figure(figsize=(10,5))
y = df_train.SalePrice
sb.set_style('whitegrid')
plt.subplot(121)
sb.distplot(y)

df_train['SalePrice_log'] = np.log(df_train.SalePrice)
y2 = df_train.SalePrice_log
plt.subplot(122)
sb.distplot(y2)
plt.show()
```
    


    
![png](output_11_1.png)
    


### Visualising the output data
- Next, taking the Sale Price data (which will bo our output variable) we plot a bar graph
- From above, we find that the data is skewed but the log-transoformed data has a much better distribution
- Such transformations help us avoid having to remove outliers.

# References:

1. https://www.educative.io/edpresso/how-to-check-if-a-key-exists-in-a-python-dictionary
2. https://towardsdatascience.com/predicting-house-prices-with-machine-learning-62d5bcd0d68f


```python

```
