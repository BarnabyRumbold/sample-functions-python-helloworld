### IMPORTS ###
import streamlit as st
import pandas as pd
import datetime
from PIL import Image

### SETUP ###


loc_path=""# for local testing
her_path="dashboard/"
path=loc_path


im = Image.open(path+"favicon-light.ico")

st.set_page_config(
    page_title="Medito Data Dashboard",
    page_icon=im)
st.title("Medito's Data Dashboard")

### DATA ###
try:
    df = pd.read_csv(path+"data.csv") #add [path + ] for heroku deployment
except FileNotFoundError:
    st.warning("Ooopps your data was not found 😥 Check if you uploaded the file." )
    df=None

if df is not None:
    rev = df[df['Review Text'].notnull()]
    rev = rev[~rev.duplicated("Review Text")]
    languages = ["french", "persian", "spanish", "hindi", "portuguese", "german", "turkish", "polish",
                 "chinese", "korean", "farsi", "dutch", "italian", "swedish"]
    lang_rev = []
    for ind in range(len(rev)):
        text = rev["Review Text"].iloc[ind]
        for word in text.split(" "):
            if word.lower() in languages:
                lang_rev.append([ind, word.lower()])

    st.subheader("Language Suggestions")
    if len(lang_rev)>0:
        st.write("This page shows the number of people suggesting to translate the app into their language.")
        out = pd.DataFrame(lang_rev).drop_duplicates()[1].value_counts()
        out = pd.DataFrame(out).reset_index().rename({"index": "Language", 1: "Number of Requests"}, axis=1)
        # CSS to inject contained in a string
        hide_table_row_index = """
                    <style>
                    tbody th {display:none}
                    .blank {display:none}
                    </style>
                    """

        # Inject CSS with Markdown
        st.markdown(hide_table_row_index, unsafe_allow_html=True)
        st.table(out)
    else:
        st.write("No languages suggested 🌞")