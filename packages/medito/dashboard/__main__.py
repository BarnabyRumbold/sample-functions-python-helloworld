### IMPORTS ###

import streamlit as st
from PIL import Image
import numpy as np
from app_store.app_store_reviews_reader import AppStoreReviewsReader
from google_play_scraper import app
from google_play_scraper import Sort, reviews_all
import pycountry
import pandas as pd

#world map
import datetime
import country_converter as coco
import folium
import requests

### SETUP ###
loc_path=""
her_path="dashboard/"
path=her_path

im = Image.open(path+"favicon-light.ico")


st.set_page_config(
    page_title="Medito Data Dashboard",page_icon=im)
st.title("Medito's Data Dashboard")
#st.image("medito_logo.png", width=150)
st.write("Check out the newest stats and visuals to see how people are liking the Medito app. ")

### DATA IMPORT ###
#file = st.file_uploader("Upload Play Store CSV here",type=["csv"])


top_countries=["es","br","us","co","de", "fr","gn","it","nl","pt"]
df=pd.DataFrame()
mapping={"us":"USA 🍔",
         "br":"Brazil ☀️",
         "es":"Spain 🐂",
         "fr":"France 🥐",
         "pt":"Portugal",
         "co":"Columbia 💃",
         "de":"Germany 🥖",
         "gn":"Guinea 🌴",
         "it":"Italy 🍝",
         "nl":"The Netherlands 🚴"}

st.write("We're now fetching GooglePlayStore reviews 🤖 Please wait...")

for lan in top_countries:
    result_temp = reviews_all(
    'meditofoundation.medito',
    sleep_milliseconds=0, # defaults to 0
    lang=lan, # defaults to 'en'
     #defaults to 'us'
    sort=Sort.MOST_RELEVANT) # defaults to Sort.MOST_RELEVANT

    #defaults to None(means all score))

    df_temp=pd.DataFrame(result_temp)
  #  st.write("We found " + str(df_temp.shape[0]) + " reviews from " + str(mapping[lan]))
    df=pd.concat([df,df_temp])

df=df.drop_duplicates(["userName","content"])

df=df.rename(columns={"score":"Star Rating", "at":"Review Submit Date and Time","content":"Review Text", "reviewCreatedVersion":"App Version Name"})

st.write("Thanks! We found "+  str(df.shape[0]) +" reviews." )
st.write("We're now fetching appstore reviews 🍎 Please wait...")


# The following code uses a package to read app store reviews into a csv when a play store csv is uploaded



## appstore
top_countries=["es","br","us","co","de", "fr","gn","it","nl","pt"]
reader_list = []
for i in top_countries:
    reader = AppStoreReviewsReader(app_id="1500780518", country=i, timeout=120.0)
    try:
        reader_list.append(reader.fetch_reviews())
    except (RuntimeError, AttributeError):
        pass

reviews_list=[]
for x in reader_list:
    if len(x)>0:
        reviews_list= reviews_list+x
data=pd.DataFrame(reviews_list)


data_apple =data.rename(columns={"version":"App Version Name","rating": "Star Rating", "title":"Review Title","content": "Review Text","date":"Review Submit Date and Time"})[["App Version Name","Star Rating","Review Title","Review Text", "country", "Review Submit Date and Time", "id"]]
st.write("We found "+str(data_apple.shape[0])+" reviews.")
data = pd.concat([data_apple, df])
data.to_csv(path+"data.csv") #add [path] for heroku deployment


st.write("Thanks for waiting, we're done here 😊 Select one of the pages in the navigation.")