![IronHack Logo](https://s3-eu-west-1.amazonaws.com/ih-materials/uploads/upload_d5c5793015fec3be28a63c4fa3dd4d55.png)

# Exploration of Poisonous Mushrooms

## Overview

PLACEHOLDER: Final Project Description will go here...

## The Data

The data set we explore here describes physical attributes and properties of over 8,000 North American mushrooms. Most significantly, each data point in the set distinguishes the mushroom as either poisonous or edible. This data was taken from OpenML.org (https://www.openml.org/d/24), extracted from "The Audubon Society Field Guide to North American Mushrooms" (1981).


## Loading the Data

In order to work with this data set, we first had to ingest it. The file containing the data was in CSV format, so after downloading it, we used the `read_csv` method to read the data into a Pandas data frame.

```python
import pandas as pd

original_mushroom_data = pd.read_csv('mushrooms.csv')
```

Next, we checked that the data set loaded correctly by checking the shape of the data frame, looking at the first few rows of the data, and evaluating the column names to ensure that they look as we would expect.

```python
original_mushroom_data.shape

(8124, 23)
```

```python
original_mushroom_data.head(5)
```

![Mushroom Data](./images/original_dataset.png)

Below, we can see the original code-values contained in each column, representing the physical attributes of each mushroom:

![Mushroom Column Values](./images/original_dataset_columns.png)


From this initial assessment, we have successfully loaded the data set and examined the shape and attributes contained within the dataset. We can now move on to evaluating the quality of the data, cleaning it, and wrangling it so that it is ready to be analyzed and modeled.

## Data Wrangling and Cleaning

The first obstacle in this dataset is that every value across all columns are stored as a codified string-value, with decoded values symbolized in the previous table. Our first task should be to create a duplicate set of data where all string-values are hot-encoded into binary (0 or 1) values, assuming that no ordinal data is available.

Before we convert this data from string to integer values, let's first remove the extraneous single quotes ' that appear in every cell to avoid problems further down the road:

```python
original_mushroom_data = original_mushroom_data.applymap(lambda x: x.replace("'", ""))
```

Next, we can check if the dataset itself is complete, by looking for missing (null or NaN) values within the dataframe:

```python
original_mushroom_data.isna().sum()
```

![Mushroom Missing Values](./images/original_dataset_missing.png)

The output from our code, shown above, tells us that the original dataset is complete with no missing values. However, you can observe that one of our column names "bruises%3F" got mangled somehow, either when importing the CSV file or originally from the source. We can fix this easily:

```python
original_mushroom_data.rename(columns={"bruises%3F": "bruises"}, inplace=True)
```

Finally, we can pass our complete and cleaned data to the Pandas "get_dummies" function in order to convert all of our string-values into boolean (numerical) data. We will call this new numerical copy of our dataframe "decoded_mushroom_data".

```python
decoded_mushroom_data = pd.get_dummies(original_mushroom_data)
```

![Decoded Mushroom Data](./images/decoded_dataset.png)


As a precaution, let's check to make sure that each column in our new "decoded_mushroom_data" is of the correct data type:

```python
decoded_mushroom_data.dtypes
```

![Decoded Mushroom Columns](./images/decoded_dataset_columns.png)


## Data Storage

Now that the data has been ingested and cleaned, we can store it in a MySQL database. To do this we first created a *mushroom* database using the `CREATE DATABASE` command in the MySQL terminal.

```sql
CREATE DATABASE mushroom;
```

We then used `pymsql` and `sqlalchemy` to connect and write the data to the database. The Python library `getpass` allows us to securely store the password used to connect to the MySQL database. The code below shows how we get that password and then use it to connect:

```python
import pymysql
import getpass
from sqlalchemy import create_engine

p = getpass.getpass(prompt='Password: ', stream=None) 
engine = create_engine('mysql+pymysql://root:'+p+'@localhost/mushroom')
```

After successful connection, we can now write both our categorical and numerical dataframes to the new MySQL database as tables called "mushroom_cat" and "mushroom_num":

```python
original_mushroom_data.to_sql('mushroom_cat', engine, if_exists='replace', index=False)
decoded_mushroom_data.to_sql('mushroom_num', engine, if_exists='replace', index=False)
```

To read the stored data back at a later date as a dataframe, we use the Pandas `read_sql_query` method:

```python
pd.read_sql_query('SELECT * FROM mushroom.mushroom_cat', engine)
pd.read_sql_query('SELECT * FROM mushroom.mushroom_num', engine)
```

We can also export the cleaned data set as a CSV file that to be imported into Tableau for exploration and reporting later on:

```python
original_mushroom_data.to_csv('./export/categorical_mushroom_data.csv', index=False)
decoded_mushroom_data.to_csv('./export/numerical_mushroom_data.csv', index=False)
```

## Data Exploration and Analysis

After cleaning and storing the data, the next steps we took were exploring and analyzing the data. Each row in the data set represents a property and each column represents attributes belonging to those properties. We looked through these attributes to determine which ones would potentially yield the most informative insights. Below are a list of those fields followed by a series of data visualizations conveying insights discovered.

* Type of Sale
* Sale condition
* Neighborhood
* Total square footage of the property
* Numbers of bedrooms and bathrooms
* Overall quality
* Overall condition
* Month and year of sale
* Sale Price
* Price per square foot

The first thing we wanted to look at was the number of sales and average sale prices by type of sale and sale condition.

![Price by Sales Type](./images/sales-avg-price-by-sale-type.png)

![Price by Sales Condition](./images/price-by-sale-condition.png)

Most of the property sales in this data set were of the *Warranty Deed - Conventional* type and under *Normal* conditions and the average price for those was in the low-to-mid $170,000's. The next most common were new construction sales, which were priced about $100,000 higher on average.

One of the most obvious drivers of sales price for a property its size, typically represented by the total number of square feet. Below is a scatter plot showing the relationship between *Total Sqft* and *Sale Price* with regression line fit to the data.

![Sqft vs. Price](./images/sqft-vs-price.png)

The R-squared for the trend line is 0.65 and the equation for the line is as follows, which can be used to set a base price for a given property given its square footage.

```text
ln(Total SQFT) = 0.63358*ln(Sale Price) + 0.184256
```

Another important factor in the value of a property is its location. Because of this, we wanted to examine how the neighborhood the property was located in impacted the value of the property. We evaluated this from three perspectives - average price by number of bedrooms, how price per square foot changes based on size of the property, and price per square foot by the number of bedrooms.

![Price by Neighborhood](./images/price-by-neighborhood.png)

![Price Per Sqft by Neighborhood and Total Sqft](./images/price-per-sqft-neighborhood-total-sqft.png)

![Price Per Sqft by Neighborhood and Bedrooms](./images/price-per-sqft-neighborhood-bedrooms.png)

From the visualizations above, we can see that there are some neighborhoods where the properties are valued higher (e.g. StoneBr, NoRidge, NridgHt, etc.) and other neighborhoods where the location hurts the value of the property (BrDale, MeadowV, Edwards, SWISU, etc.). This also lets you see the neighborhoods where someone may be able to get the most property for their money. For example, larger (6-8 bedroom) properties in the NAmes, Sawyer, and SWISU neighborhoods have significantly lower value per square foot than smaller properties in those areas.

There were a few other perspectives we wanted to look at the data from as well. We noticed that there were fields representing overall condition and overall quality and wanted to take a closer look at the differences between them and the impact they had on sales price.

![Condition vs. Quality](./images/condition-vs-quality.png)

It looks like most properties were rated a 5 for condition and were rated somewhere in the 5-8 range for quality. It looks like the quality rating for these properties is especially important as we see an typical increase of between 20% and 35% in average sales price for every point on the scale a property earns.

We also wanted to take a look at how price varies based on the number of bedrooms and bathrooms a property has. From the visualization below, it looks as though half bathrooms are valued pretty highly - especially in 4 bedroom properties (between 45% and 72% increase in price).

![Bedroom vs. Bathroom](./images/bedroom-vs-bathroom.png)

Finally, we also wanted to look at how average prices have typically changed through out the year and from one year to the next. The visualization below shows an area chart for average price by month, stacked by year.

![Avg Price by Month and Year](./images/month-year.png)

From this, we can see that average sale prices are relatively consistent throughout the year, aside from a small bump in September. Additionally, it looks as though prices took a slight downturn over the last couple years (2008 and 2009) - down about 6% from the previous two years.

## Feature Selection

Once we had a better understanding of the data and some insight as to how different property attributes impact pricing, we proceeded to prep our data set for modeling. To do this, we narrowed down our feature set to just those that we explored and analyzed so that we could see how well those features can inform a machine learning model whose goal is to predict the sale price of a given property.

```python
features = ['Neighborhood', 'OverallQual', 'OverallCond', 
            'FullBath', 'HalfBath', 'BedroomAbvGr', 
            'MoSold', 'YrSold', 'SaleType', 'SaleCondition', 
            'SalePrice', 'Total Sqft', 'Price Per Sqft']

selected = data[features]
```

Next, we transformed these features so that all categorical variables were one-hot encoded.

```python
transformed = pd.get_dummies(selected)
```

## Model Training and Evaluation

We decided to compare several supervised learning regression models via k-fold cross validation to see which ones would best be able to predict the sale price of a property. To do this, we imported a variety of regression models from Scikit-learn and wrote a `compare_models` function that would train and evaluate multiple regressor models (stored in a dictionary) and compute cross validation scores for each so that we can compare their performance.

```python
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor

regressors = {'K-Nearest Neighbor': KNeighborsRegressor(n_neighbors=3),
              'Decision Tree': DecisionTreeRegressor(),
              'Ridge Regression': Ridge(),
              'Random Forest': RandomForestRegressor(),
              'Gradient Boosting': GradientBoostingRegressor(), 
              'AdaBoost': AdaBoostRegressor(),
              'Bagging Regressor': BaggingRegressor()}

def compare_models(x, y, model_dict, folds=3):
    results = []
    
    for name, model in model_dict.items():
        scores = cross_val_score(model, x, y, cv=folds)
        stats = [name, scores.mean(), min(scores), max(scores), scores.std(), pd.Series(scores).mad()]
        results.append(stats)
    
    df = pd.DataFrame(results, columns = ['Model', 'Mean', 'Min', 'Max','Std', 'Mad'])
    df = df.sort_values('Mean', ascending = False)
    return df
```

Next, we designated the sale price field as our target (y) variable and the rest of the fields (with the exception of price per sqft since the sale price is used to calculate it) as our independent variable (x) set.

```python
y = transformed['SalePrice']
x = transformed.drop(['SalePrice', 'Price Per Sqft'], axis=1)
```

We then ran our model comparison function using 5-fold cross validation and obtained the following results.

```python
compare_models(x, y, regressors, 5)
```

![Avg Price by Month and Year](./images/model-comparison.png)

It looks like the Gradient Boosting model performed the best with an average score of 0.85. We then used that algorithm to train a model on the entire data set and pickle it for future use.

```python
import pickle

model = GradientBoostingRegressor()
model.fit(x, y)
pickle.dump(model, open('housing_price_model.pkl', 'wb'))
```

## Conclusion

In this demo final project, we have analyzed a data set consisting of over 1,400 properties, their attributes, and their sale prices. We have followed the steps of the data analysis workflow, starting with data ingestion, wrangling and cleaning, and exploration and analysis before moving on to the the machine learning workflow consisting of feature selection/engineering, model selection, and model evaluation. In the end, we were able to predict housing prices with a respectable level of accuracy and we also derived insights about how factors such as neighborhood, square footage, and quality of the property affect the sale price.
