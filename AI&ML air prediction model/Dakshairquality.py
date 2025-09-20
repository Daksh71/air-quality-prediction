import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from warnings import filterwarnings
filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('C:\\VS Code\\Python\\air quality data-Copy1.csv')

# Display the first few rows of the dataset
print(df.head())

# Check the number of rows and columns in the dataset
print("Dataset Shape:", df.shape)

# Get an overview of the dataset
print("Dataset Information:")
print(df.info())

# Check for missing values in the dataset
print("Missing Values:")
print(df.isnull().sum())

# Check for duplicate values in the dataset
print("Duplicate Values:", df.duplicated().sum())

# Drop rows where the 'AQI' column has missing values
df = df.dropna(subset=['AQI'])

# Check for missing values again after dropping rows
print("Missing Values after dropping rows:")
print(df.isnull().sum().sort_values(ascending=False))

# Check the shape of the dataset after dropping rows
print("Dataset Shape after dropping rows:", df.shape)

# Get summary statistics for the dataset
print("Summary Statistics:")
print(df.describe().T)

# Calculate the percentage of missing values in the dataset
null_values_percentage = (df.isnull().sum()/df.isnull().count()*100).sort_values(ascending=False)
print("Percentage of Missing Values:")
print(null_values_percentage)

#Week 2
df['Xylene'].plot(kind='hist',figsize=(10,5))
plt.legend()
plt.show()

df['PM10'].plot(kind='hist',figsize=(10,5))
plt.legend()
plt.show()

df['NH3'].plot(kind='hist',figsize=(10,5))
plt.legend()
plt.show()

df['Toluene'].plot(kind='hist',figsize=(10,5))
plt.legend()
plt.show()

df['Benzene'].plot(kind='hist',figsize=(10,5))
plt.legend()
plt.show()

df['NOx'].plot(kind='hist',figsize=(10,5))
plt.legend()
plt.show()

df['O3'].plot(kind='hist',figsize=(10,5))
plt.legend()
plt.show()

df['PM2.5'].plot(kind='hist',figsize=(10,5))
plt.legend()
plt.show()

df['SO2'].plot(kind='hist',figsize=(10,5))
plt.legend()
plt.show()

df['CO'].plot(kind='hist',figsize=(10,5))
plt.legend()
plt.show()

df['NO2'].plot(kind='hist',figsize=(10,5))
plt.legend()
plt.show()

df['NO'].plot(kind='hist',figsize=(10,5))
plt.legend()
plt.show()

df['AQI'].plot(kind='hist',figsize=(10,5))
plt.legend()
plt.show()

# distribution of aqi from 2015-2020
sns.displot(df, x="AQI", color="purple")
plt.show()

sns.set(style="darkgrid")
graph=sns.catplot(x="City",kind="count",data=df,height=5,aspect=3)
graph.set_xticklabels(rotation=90)

sns.set(style="darkgrid")
graph=sns.catplot(x="City",kind="count",data=df,col="AQI_Bucket",col_wrap=2,height=3.5,aspect=3)
graph.set_xticklabels(rotation=90)

graph1=sns.catplot(x="City",y="PM2.5",kind="box",data=df,height=5,aspect=3)
graph1.set_xticklabels(rotation=90)

graph2=sns.catplot(x="City",y="NO2",kind="box",data=df,height=5,aspect=3)
graph2.set_xticklabels(rotation=90)

graph3=sns.catplot(x="City",y="O3",data=df,kind="box",height=5,aspect=3)
graph3.set_xticklabels(rotation=90)

graph4=sns.catplot(x="City",y="SO2",data=df,kind="box",height=5,aspect=3)
graph4.set_xticklabels(rotation=90)

graph5=sns.catplot(data=df,kind="box",x="City",y="NOx",height=6,aspect=3)
graph5.set_xticklabels(rotation=90)

graph6=sns.catplot(data=df,kind="box",x="City",y="NO",height=6,aspect=3)
graph6.set_xticklabels(rotation=90)

graph7=sns.catplot(x="AQI_Bucket",data=df,kind="count",height=6,aspect=3)
graph7.set_xticklabels(rotation=90)

# Checking all null values

df.isnull().sum().sort_values(ascending=False)

# higher null values present in PM10 followed by NH3

df.describe().loc["mean"]

df = df.replace({

"PM2.5" : {np.nan:67.476613},
"PM10" :{np.nan:118.454435},
"NO": {np.nan:17.622421},
"NO2": {np.nan:28.978391},
"NOx": {np.nan:32.289012},
"NH3": {np.nan:23.848366},
"CO":  {np.nan:2.345267},
"SO2": {np.nan:34.912885},
"O3": {np.nan:38.320547},
"Benzene": {np.nan:3.458668},
"Toluene": {np.nan:9.525714},
"Xylene": {np.nan:3.588683}})

df.isnull().sum()

graph=sns.catplot(x="AQI_Bucket",data=df,kind="count",height=6,aspect=3)
graph.set_xticklabels(rotation=90)

df = df.drop(["AQI_Bucket"], axis=1)

df.head()

sns.boxplot(data=df[[ 'PM2.5', 'PM10']])

sns.boxplot(data=df[[ 'NO', 'NO2', 'NOx','NH3']]) 

sns.boxplot(data=df[[ 'O3', 'CO', 'SO2']])

sns.boxplot(data=df[[ 'Benzene', 'Toluene', 'Xylene']])

#
#
#
# This function takes a DataFrame as a parameter and identifies outliers for numeric columns in the DataFrame. 
#It replaces these outliers with the corresponding quartile values ​​(Q1 or Q3). Outliers are identified using the interquartile range (IQR).
def replace_outliers_with_quartiles(df):
    
    for column in df.select_dtypes(include=['number']).columns: # Used to cycle through all numeric columns in the DataFrame.
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        # To identify outliers, lower and upper limits are calculated and values ​​outside these limits are considered outliers.
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        # For each column, we identify outliers and replace them with Q1 or Q3. We do this using a lambda function.
        #If the value is less than the lower bound, it is replaced with Q1. If it is greater than the upper bound, 
        #it is replaced with Q3. In the last case, the value is not changed and remains the same.
        df[column] = df[column].apply(
            lambda x: Q1 if x < lower_bound else (Q3 if x > upper_bound else x)
        )
    
    return df 

df = replace_outliers_with_quartiles(df)

df.describe().T

sns.boxplot(data=df[[ 'PM2.5', 'PM10']])

sns.boxplot(data=df[[ 'NO', 'NO2', 'NOx','NH3']])

sns.boxplot(data=df[[ 'O3', 'CO', 'SO2']])

sns.boxplot(data=df[[ 'Benzene', 'Toluene', 'Xylene']])

# distribution of aqi from 2015-2020
sns.displot(df, x="AQI", color="red")
plt.show()

df1=df.drop(columns=['City'])
numeric_df = df1.select_dtypes(include=['number'])
correlation_matrix = numeric_df.corr()
sns.heatmap(correlation_matrix, annot=True)
plt.figure(figsize=(12, 8)) 

plt.show() 


df.head()

df

# Dropping unnecessary columns
df.drop(['Date'],axis=1,inplace=True)        # no need of this feature
df.drop(['City'],axis=1,inplace=True)        # as we are going to calculate based on other parameters not on the loaction so we drop this


pd

from sklearn.preprocessing import StandardScaler
df1 = StandardScaler().fit_transform(df)

df1

df = pd.DataFrame(df1,columns = df.columns)

df

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

# Data Preparation for Modeling
x=df[["PM2.5","PM10","NO","NO2","NOx","NH3","CO","SO2","O3","Benzene","Toluene","Xylene"]]
y=df["AQI"]

x.head()

y.head()

X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=70)
print(X_train.shape,X_test.shape,Y_train.shape,Y_test.shape)
# splitting the data into training and testing data


model=LinearRegression()
model.fit(X_train,Y_train)

#predicting train
train_pred=model.predict(X_train)
#predicting on test
test_pred=model.predict(X_test)


RMSE_train=(np.sqrt(metrics.mean_squared_error(Y_train,train_pred)))
RMSE_test=(np.sqrt(metrics.mean_squared_error(Y_test,test_pred)))
print("RMSE TrainingData = ",str(RMSE_train))
print("RMSE TestData = ",str(RMSE_test))
print('-'*50)
print('RSquared value on train:',model.score(X_train, Y_train))
print('RSquared value on test:',model.score(X_test, Y_test))

KNN = KNeighborsRegressor()
KNN.fit(X_train,Y_train)

#predicting train
train_pred=model.predict(X_train)
#predicting on test
test_pred=model.predict(X_test)

RMSE_train=(np.sqrt(metrics.mean_squared_error(Y_train,train_pred)))
RMSE_test=(np.sqrt(metrics.mean_squared_error(Y_test,test_pred)))
print("RMSE TrainingData = ",str(RMSE_train))
print("RMSE TestData = ",str(RMSE_test))
print('-'*50)
print('RSquared value on train:',model.score(X_train, Y_train))
print('RSquared value on test:',model.score(X_test, Y_test))

DT=DecisionTreeRegressor()
DT.fit(X_train,Y_train)

#predicting train
train_preds=DT.predict(X_train)
#predicting on test
test_preds=DT.predict(X_test)

RMSE_train=(np.sqrt(metrics.mean_squared_error(Y_train,train_preds)))
RMSE_test=(np.sqrt(metrics.mean_squared_error(Y_test,test_preds)))
print("RMSE TrainingData = ",str(RMSE_train))
print("RMSE TestData = ",str(RMSE_test))
print('-'*50)
print('RSquared value on train:',DT.score(X_train, Y_train))
print('RSquared value on test:',DT.score(X_test, Y_test))

RF=RandomForestRegressor()
RF.fit(X_train,Y_train)

#predicting train
train_preds1=RF.predict(X_train)
#predicting on test
test_preds1=RF.predict(X_test)

RMSE_train=(np.sqrt(metrics.mean_squared_error(Y_train,train_preds1)))
RMSE_test=(np.sqrt(metrics.mean_squared_error(Y_test,test_preds1)))
print("RMSE TrainingData = ",str(RMSE_train))
print("RMSE TestData = ",str(RMSE_test))
print('-'*50)
print('RSquared value on train:',RF.score(X_train, Y_train))
print('RSquared value on test:',RF.score(X_test, Y_test))


