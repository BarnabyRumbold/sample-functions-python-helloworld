### IMOPRTS ###
from pickletools import read_stringnl_noescape
import streamlit as st
import pandas as pd
from PIL import Image


### SETUP ###
#path="../"# for local testing
loc_path=""# for local testing
her_path="dashboard/"
path=loc_path

im = Image.open(path+"favicon-light.ico")


st.set_page_config(
    page_title="Medito Data Dashboard",
    page_icon=im)
st.title("Medito's Data Dashboard")


st.subheader("Average Star-Rating by Version")
st.write(
    'Average Star-Rating by version considering app versions that have been rated at least 5 times. Bars in red are app versions that have an average score of 4.5 or lower.')

try:
    df = pd.read_csv(path + "data.csv") #add [path + ] for heroku deployment
except FileNotFoundError:
    st.warning("Ooopps your data was not found 😥 Check if you uploaded the file." )
    df=None

if df is not None:

    THRESHOLD = 5
    df_versions = df[["App Version Name", "Star Rating"]].dropna(0, "any")

# create second visual data 
    data2 = df_versions
    df_versions = df_versions.groupby("App Version Name").agg({'Star Rating': ['mean', 'count']}).sort_index()
    df_versions = df_versions[df_versions["Star Rating"]["count"] > THRESHOLD]
    #st.bar_chart(df_versions["Star Rating"]["mean"])

# create second df = data2
    import matplotlib.pyplot as plt
    import altair as alt
    data2 = data2.groupby("App Version Name", as_index=False).mean().sort_index().round(2)
    data2 = data2.rename(columns={'Star Rating': 'Rating'})

# Create bar

    bar = alt.Chart(data2).mark_bar().encode(
        x=alt.X('App Version Name'),
        y=alt.Y('Rating'),
        tooltip="Rating",
        color = alt.condition(
        alt.datum.Rating <= 4.5,  # If the average review is less than 4.5 this returns True,
        alt.value('red'),     # which sets the bar red.
        alt.value('grey'))).configure_view(strokeOpacity=0).configure_axis(domainOpacity=0)
# Display bar

    final_bar = (bar).properties(
        height = 500,
        width = 600).configure_mark(opacity=1).configure_view(strokeOpacity=0)
    final_bar

