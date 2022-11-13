import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import datetime
import country_converter as coco
import folium
import requests
import pycountry
from folium import Tooltip
from streamlit_folium import st_folium

### SETUP ###

loc_path=""# for local testing
her_path="dashboard/"
path=loc_path


im = Image.open(path+"favicon-light.ico")


st.set_page_config(
    page_title="Apple Store World Map",
    page_icon=im)
st.title("Apple Store World Map")
#st.image("medito_logo.png", width=150)
st.write("This page uses Apple Store Data to show graphically the total downloads by country, from the data you upload. Upload a CSV, then roll over for downloads by territory.")

### DATA IMPORT ###
st.header("Upload Apple Store CSV")
file = st.file_uploader("Upload CSV here",type="csv")

if file is not None:
    data = pd.read_csv(file, sep=',', header=3)

# Drop date column
    data = data.drop('Date', axis=1)
# Clean column titles for viz application
    data.columns = data.columns.str.strip("Downloads")

# Drop columns with no values to allow sum
    data = data.loc[:, ~(data == '-').any()]

# Create series of sum total for each column
    data = data.sum()

# Convert series back to data frame
    data = data.to_frame().reset_index()
    data.columns = ['Country', 'Count']

# Stripping 'Total' from 'Country' column to allow country codes addition
    data['Country'] = data['Country'].str.replace(r'Total', '')
    data['Country'] = data['Country'].str.strip(" ")

# Data type needs to be changed to allow country code change, convert data types to integer
    data["Count"] = data["Count"].astype(int)

# Change country to country code - creating a list
    countries = []
    for i in data["Country"]:
        countries.append(i)

# Converting country to country code and fixing errors

  #  i = countries.index('enmark')
  #  countries = countries[:i] + ['Denmark'] + countries[i + 1:]
 #   j = countries.index('minican Republic  ')
 #   countries = countries[:j] + ["Dominican Republic"] + countries[j + 1:]



# Read in GeoJson. This is the data that allows country code to be used for world map viz
    geojson_url = 'https://raw.githubusercontent.com/python-visualization/folium/main/examples/data/world-countries.json'
    response = requests.get(geojson_url)
    geojson = response.json()

    country_codes = coco.convert(names=countries, to='ISO3')
    data['Country_Code'] = country_codes


# Create map
    map = folium.Map(location=[20, 10], zoom_start=2)

#Upload data to map

    for s in geojson['features']:
        if len(data[data["Country_Code"]==s["id"]])>0:
            s['properties']['Count']=str(data[data["Country_Code"]==s["id"]]["Count"].iloc[0])
        else:
            s['properties']['Count']=str(0)
    cp=folium.Choropleth(geo_data=geojson,
                  data=data,
                  columns=["Country_Code", "Count"],
                  key_on='feature.id',
                  fill_color="YlOrRd",
                  nan_fill_color = "#ffffb2", # Use white color if there is no data available for the county
                  fill_opacity=0.7,
                  line_opacity=0.2,
                  legend_name="Number of Medito downloads in each country this month.",  # title of the legend
                  highlight=True,
                  line_color='black',
                  tooltip=folium.features.GeoJsonTooltip(['Count'])
                  ).add_to(map)
    # for s in cp.geojson.data['features']:
    #     if len(data[data["Country_Code"]==s["id"]])>0:
    #         s['properties']['Count']=data[data["Country_Code"]==s["id"]]["Count"].iloc[0]
    #     else:
    #         s['properties']['Count']=0
    folium.GeoJsonTooltip(["name","Count"]).add_to(cp.geojson)

    st_data = st_folium(map, width = 725)

else:
    pass
