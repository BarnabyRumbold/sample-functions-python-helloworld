# This uses review data as an indicator of engagement to predict future usage for coming server cost. 

from re import I
from sqlite3 import Timestamp
import pandas as pd
import streamlit as st
import datetime
from PIL import Image
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
from sklearn.model_selection import train_test_split
import numpy as np
from scalecast.Forecaster import Forecaster
from pmdarima import auto_arima
import seaborn as sns
from statsmodels.tsa.arima_model import ARIMA
import statsmodels as sts

loc_path=""# for local testing
her_path="dashboard/"
path=loc_path
im = Image.open(path+"favicon-light.ico")

try:
    input_df = pd.read_csv(path+"data.csv")
except FileNotFoundError:
    st.warning("Ooopps your data was not found 😥 Check if you uploaded the file." )
    input_df=None

if input_df is not None:

    input_df["Date"] = pd.to_datetime(input_df['Review Submit Date and Time'], utc=True)
    input_df['datetime'] = input_df['Date'].apply(lambda x: datetime.date(x.year, x.month, x.day))
    STARTDATE = input_df['datetime'].min()
    ENDDATE = input_df['datetime'].max()

# First group by month and count the number of rows in each month and then plot this to show time series of number of users.
    st.header("Time Series of Number of Reviews")
    st.write("A rough indication of current userbase and growth over time")

# Create "Count" column indicating if a review was left
    input_df["Count"] = 1
    input_df["Date"] = input_df.Date.dt.strftime('%Y-%m-%d')
    input_df["Date"] = input_df["Date"].astype("datetime64[ns]")
    print(len(input_df["Date"]))

# Create new df of dates in date range of data
    min = input_df['Date'].min()
    max = input_df["Date"].max()
    all_dates = pd.date_range(min,max,freq='d')
    date_df = pd.DataFrame(all_dates, columns=["Date"])

# Create boolean mask where rows where no reviews were left. 
    u = date_df.merge(input_df, how='outer', indicator='bool', on="Date")
    u['bool'] = u['bool'] == 'both'
    ml_df = u[['Date', 'bool']].copy()
    print(ml_df["bool"].value_counts())

# Replace boolean values 
    ml_df["Count"] = ml_df["bool"].astype(int)
    ml_df = ml_df[["Date", "Count"]]
    print(ml_df)

# Let's groupy by month and plot this 


# DO BY WEEK AND WHOLE RANGE OF DATA
    x = ml_df.groupby(ml_df.Date.dt.month)['Count'].sum()
    x = pd.DataFrame(x)
    print(x)
    st.line_chart(x)
    
# Let's build the model
    f = Forecaster(y=x['Count'],current_dates=x.index, future_dates=6)
    f.generate_future_dates(12) # 12-month forecast horizon
    f.set_test_length(.2) # 20% test set
    f.set_estimator('arima') # set arima
    f.manual_forecast(call_me='arima1', dynamic_testing=True) # forecast with arima
    f.plot(level=True) # view test results
    plt.title('ARIMA Test-Set Performance',size=14)
    st.pyplot(plt)