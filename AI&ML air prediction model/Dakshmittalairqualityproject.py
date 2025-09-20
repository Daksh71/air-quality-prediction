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

def load_data(filepath):
    """Load dataset from a CSV file."""
    try:
        with open(filepath, 'r') as file:
            df = pd.read_csv(r"C:\VS Code\Python\air quality data-Copy1.csv")
            print("Data loaded successfully.")
            return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def inspect_data(df):
    """Display basic information about the dataset."""
    print(df.head())
    print("Dataset Shape:", df.shape)
    print("Dataset Information:")
    print(df.info())
    print("Missing Values:")
    print(df.isnull().sum())
    print("Duplicate Values:", df.duplicated().sum())

def clean_data(df):
    """Clean the dataset by dropping rows with missing AQI values."""
    df.dropna(subset=['AQI'], inplace=True)
    print("Missing Values after dropping rows:")
    print(df.isnull().sum().sort_values(ascending=False))
    print("Dataset Shape after dropping rows:", df.shape)

def summarize_data(df):
    """Display summary statistics and visualize missing values."""
    print("Summary Statistics:")
    print(df.describe().T)
    null_values_percentage = (df.isnull().sum()/df.isnull().count()*100).sort_values(ascending=False)
    print("Percentage of Missing Values:")
    print(null_values_percentage)
    
    # Visualizing missing values
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title('Missing Values Heatmap')
    plt.show()

# Main execution
if __name__ == "__main__":
    filepath = 'C:\\VS Code\\Python\\air quality data-Copy1.csv'
    df = load_data(filepath)
    
    if df is not None:
        inspect_data(df)
        clean_data(df)
        summarize_data(df)

import matplotlib.pyplot as plt
import seaborn as sns

# Set the style for seaborn
sns.set(style="darkgrid")

# Function to plot histograms
def plot_histogram(data, column):
    plt.figure(figsize=(10, 5))
    data[column].plot(kind='hist', edgecolor='black')
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()

# List of columns to plot histograms
columns_to_plot = ['Xylene', 'PM10', 'NH3', 'Toluene', 'Benzene', 
                   'NOx', 'O3', 'PM2.5', 'SO2', 'CO', 'NO2', 'NO', 'AQI']

# Loop through columns to create histograms
for column in columns_to_plot:
    plot_histogram(df, column)

# Distribution of AQI from 2015-2020
plt.figure(figsize=(10, 5))
sns.displot(df, x="AQI", color="purple")
plt.title('Distribution of AQI from 2015-2020')
plt.show()

# Count plots for City
graph = sns.catplot(x="City", kind="count", data=df, height=5, aspect=3)
graph.set_xticklabels(rotation=90)
plt.title('Count of Entries by City')
plt.show()

# Count plots for City by AQI Bucket
graph = sns.catplot(x="City", kind="count", data=df, col="AQI_Bucket", col_wrap=2, height=3.5, aspect=3)
graph.set_xticklabels(rotation=90)
plt.title('Count of Entries by City and AQI Bucket')
plt.show()

# Box plots for different pollutants
pollutants = ['PM2.5', 'NO2', 'O3', 'SO2', 'NOx', 'NO']
for pollutant in pollutants:
    graph = sns.catplot(x="City", y=pollutant, kind="box", data=df, height=5, aspect=3)
    graph.set_xticklabels(rotation=90)
    plt.title(f'Box Plot of {pollutant} by City')
    plt.show()

# Count plot for AQI Bucket
graph = sns.catplot(x="AQI_Bucket", data=df, kind="count", height=6, aspect=3)
graph.set_xticklabels(rotation=90)
plt.title('Count of Entries by AQI Bucket')
plt.show()

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

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

# Data Preparation for Modeling
x = df[["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene"]]
y = df["AQI"]

# Splitting the data into training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=70)
print(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")

# Function to evaluate models
def evaluate_model(model, X_train, Y_train, X_test, Y_test):
    model.fit(X_train, Y_train)
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    RMSE_train = np.sqrt(mean_squared_error(Y_train, train_pred))
    RMSE_test = np.sqrt(mean_squared_error(Y_test, test_pred))
    
    print(f"Model: {model.__class__.__name__}")
    print(f"RMSE Training Data: {RMSE_train:.2f}")
    print(f"RMSE Test Data: {RMSE_test:.2f}")
    print(f'R-Squared value on train: {model.score(X_train, Y_train):.2f}')
    print(f'R-Squared value on test: {model.score(X_test, Y_test):.2f}')
    print('-' * 50)

# Evaluate different models
models = {
    "Linear Regression": LinearRegression(),
    "K-Neighbors Regressor": KNeighborsRegressor(),
    "Decision Tree Regressor": DecisionTreeRegressor(),
    "Random Forest Regressor": RandomForestRegressor()
}

for model_name, model in models.items():
    evaluate_model(model, X_train, Y_train, X_test, Y_test)

