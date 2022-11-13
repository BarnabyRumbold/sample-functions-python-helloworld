### IMPORTS ###
import streamlit as st
import pandas as pd
import datetime
from PIL import Image
import altair as alt
import plotly.graph_objs as go


### SETUP ###
loc_path=""# for local testing
her_path="dashboard/"
path=loc_path

im = Image.open(path +"favicon-light.ico")

st.set_page_config(
    page_title="Medito Data Dashboard",
    page_icon=im)
st.title("Medito's Data Dashboard")

### DATA ###
try:
    df = pd.read_csv(path +"data.csv", date_parser=True)
except FileNotFoundError:
    st.warning("Ooopps your data was not found 😥 Check if you uploaded the file." )
    df=None



if df is not None:
    df = df[["Star Rating", "Review Submit Date and Time"]].sort_values("Review Submit Date and Time")
    df['Review Submit Date and Time']=pd.to_datetime(df['Review Submit Date and Time'], utc=True)
    df["Review Submit Date and Time"]= df["Review Submit Date and Time"].apply(lambda x: datetime.date(x.year, x.month, x.day))
    df["Review Submit Date and Time"] = pd.DatetimeIndex(df["Review Submit Date and Time"])#.to_period("M")


    df.set_index("Review Submit Date and Time")
  #  df["Review Submit Date and Time"] = df["Review Submit Date and Time"].apply(lambda x: datetime.date(1900, x, 1))
    g_df = df.groupby("Review Submit Date and Time", as_index=False).mean()
    monthly_avg = alt.Chart(df, height=400, width=600).mark_bar().encode(
        x=alt.X("yearmonth(Review Submit Date and Time)",type="temporal"),
        y=alt.Y("Star Rating", aggregate="average",scale=alt.Scale(domain=[4.5, 5]) ),
        tooltip=alt.Y("Star Rating", aggregate="average"),
        color = alt.value("orange"))
    # Display bar
  #  line = alt.Chart().mark_rule().encode(y=4.5)

    final_bar = (monthly_avg).properties(height=400, width=600).configure_mark(opacity=1).configure_view(strokeOpacity=0)
    st.subheader("Average Star Rating by Month")
    st.write("Roll over for average rating")
    monthly_avg


    monthly_stacked = alt.Chart(df, height=400, width=600).mark_bar().encode(
        x=alt.X("yearmonth(Review Submit Date and Time)",type="temporal"),
        y=alt.Y("count(Star Rating)", type="quantitative" ),
        tooltip=alt.Y("Star Rating", aggregate="count"),
        color=alt.Color("Star Rating:O", scale=alt.Scale(scheme='plasma')))

    monthly_stacked = (monthly_stacked).properties(height=400, width=600).configure_mark(opacity=1).configure_view(strokeOpacity=0)

    st.subheader("Total Count and Breakdown of Star Reviews")
    st.write("Roll over for total count by star rating")
    monthly_stacked


    ###########################################################

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

    ############### Calculations for 'Analysis per Star Rating' in Pages: 'Reviews Rating  X' ###############
    # TODO: make sure that you only work with reviews in english and that you display this as a warning/pop-up

    my_reviews_df = pd.DataFrame()
    my_reviews_df['date'] = input_df['datetime']
    my_reviews_df['year'] = input_df['datetime'].apply(lambda x: x.year)
    my_reviews_df['month'] = input_df['datetime'].apply(lambda x: x.month)
    my_reviews_df['rating'] = input_df['Star Rating']
    my_reviews_df['for_version'] = input_df['App Version Name']
    #my_reviews_df['dev_response'] = input_df['Developer Reply Date and Time']
    my_reviews_df['user_review'] = input_df['Review Text']
    my_reviews_df.loc[my_reviews_df['user_review'].isnull(), 'user_review'] = ''
    colours = ["#009700", "#009790", "#FFFF5C", "#FF8A33", "#FF3F31"]
    axis_style = dict(
        ticks="outside",
        fixedrange=True,
        showline=False,
        # showgrid=False,
        showticklabels=True,
        ticklen=10,
        tickcolor='white',
        tickfont=dict(family='Arial', size=12, color='rgb(82, 82, 82)', ),
    )
    hover_label_style = dict(bgcolor="white", font_size=15)
    margin_style = dict(t=0, b=0)
    reviews_bins = [(my_reviews_df.rating == 1).sum(), (my_reviews_df.rating == 2).sum(),
                    (my_reviews_df.rating == 3).sum(), (my_reviews_df.rating == 4).sum(),
                    (my_reviews_df.rating == 5).sum()]

    fig = go.Figure()
    data = [
        go.Bar(x=reviews_bins, y=[1, 2, 3, 4, 5], name='', orientation='h',
               marker={"color": colours[::-1], }, opacity=0.8, hoverinfo='skip',
               text=reviews_bins, textposition='outside', cliponaxis=False
               )
    ]

    fig = go.Figure(data=data)

    fig.update_layout(
        height=400, width=600,
        xaxis=dict(showticklabels=False, ticks="", fixedrange=True, ),
        yaxis={**axis_style, **{'tick0': 0, 'dtick': 1}},
        margin=margin_style,
        plot_bgcolor='white',
    )

    overall_df = my_reviews_df  # TODO: (nice to have) get rid of this overall_df if it is really just a copy of my_reviews_df

    overall_df.dropna(subset=["user_review"], inplace=True)
    overall_review = ' '.join(overall_df[['date', 'user_review']].user_review).lower()
    # overall_df['date'] = overall_df.date.map(lambda x: x.date())


    # from monkeylearn import MonkeyLearn

    overall_rating_df = overall_df.groupby('rating').count().reset_index()[['rating', 'user_review']]
    total_reviews = overall_rating_df.user_review.sum()
    overall_rating_df['percentage of total'] = overall_rating_df.user_review.map(
        lambda x: round((x / total_reviews) * 100, 2))

    axis_style = dict(
        fixedrange=True,
        showline=True,
        showgrid=False,
        showticklabels=True,
        linecolor='rgb(204, 204, 204)',
        linewidth=2,
        ticks='outside',
        tickfont=dict(family='Arial', size=12, color='rgb(82, 82, 82)', ),
    )
    hover_label_style = dict(bgcolor="white", font_size=15)
    margin_style = dict(l=100, r=20, t=100, )
    ls_data = []

    for i in range(1, 6):
        ls_data.append(
            go.Scatter(x=overall_df[overall_df.rating == i].groupby('date').count().reset_index().date.tolist(),
                       y=overall_df[overall_df.rating == i].groupby('date').count().rating.tolist(),

                       name=i, mode='lines+markers', marker=dict(size=1, line={'width': 2})
                       ))
    fig_overall = go.Figure(data=ls_data)
    fig_overall.update_layout(
        hoverlabel=hover_label_style,
        hovermode='closest',
        # hovermode='x unified',
        xaxis_title="Time Period",
        yaxis_title="Total",
        xaxis=axis_style,
        yaxis=axis_style,
        margin=margin_style,
        showlegend=True,
        plot_bgcolor='white',
    )

    st.subheader("Interactive Timeline of All Review Ratings")
    st.write("Roll over for date and use legend to select a particular rating score")
    fig_overall = fig_overall.update_layout(height=500,width=800)
    st.plotly_chart(fig_overall)
