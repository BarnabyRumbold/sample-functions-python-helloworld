import pandas as pd
import streamlit as st
from PIL import Image
import datetime
import numpy as np
import pandas as pd
import datetime
import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import RegexpTokenizer
from string import punctuation as punc
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords as sw
import plotly.graph_objs as go
import re


### SETUP ###
loc_path=""# for local testing
her_path="dashboard/"
path=loc_path
im = Image.open(path+"favicon-light.ico")

st.set_page_config(
    page_title="Medito Data Dashboard",
    page_icon=im)
st.title("Medito's Data Dashboard")


st.subheader("Analysis by Star Rating")
st.write('This page provides language analysis of reviews, looking at the most common words used, filtered by star rating. It also provides review text for a more in depth insight.')

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

    #rev_res = my_reviews_df.groupby('rating').count()[['user_review', 'dev_response']].reset_index()
    stopwords = set(sw.words('english'))
    stopwords.update(  # TODO: fix it cause some words like 'meditation'
        {'sometimes', 'get', 'i\'m', 'good', 'something', 'give', 'hope', 'that\'s', 'that', 'well', 'please', 'plz',
         'help', 'also', 'u', 'app', 'meditation', 'meditate', 'love', 'like', "i've", 'one', 'really', 'apps', 'thank',
         'amaze', 'make', 'much', 'huge', 'strong', 'great', 'medito', 'meditation', 'lot', "can't", 'make', 'would',
         'could'})
    stopwords.difference_update(
        {"aren't", "couldn't", "doesn't", "don't", "hadn't", "haven't", "isn't", "shouldn't", "weren't", "won't",
         "wouldn't"})
    lemmatizer = WordNetLemmatizer()


    def create_word_graph(voc):
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
        data = [
            go.Bar(x=[tup[0] for tup in voc.most_common(20)], y=[tup[1] for tup in voc.most_common(20)],
                   name='', hoverinfo='skip', marker={"color": "#63C9B5"},
                   text=[tup[1] for tup in voc.most_common(20)], textposition='outside', cliponaxis=False
                   )
        ]
        fig = go.Figure(data=data)
        fig.update_layout(
            height=400,
            title_text="Top words used",
            hoverlabel=hover_label_style,
            hovermode='x',
            xaxis_title="words",
            yaxis_title="frequency",
            xaxis=axis_style,
            yaxis=axis_style,
            margin=margin_style,
            showlegend=False,
            plot_bgcolor='white',
        )
        return fig


    ### Add your keywords to filter out reviews
    keywords = []  # ['otp']

    ls_dic_review = []
    ls_fig = []

    tokenizer = RegexpTokenizer("[a-z']{3,}")

    for curr_rating in range(1,6):
        review_df = my_reviews_df[my_reviews_df.rating == curr_rating]  # new dataframe for specific star rating
        dic_review = {}
        dic_word = {}

        review = ' '.join(review_df.user_review).lower()
        token = tokenizer.tokenize(review)
        token = [lemmatizer.lemmatize(lemmatizer.lemmatize(word, pos='v'), pos='n') for word in token if
                 word.lower() not in stopwords]
        voc = nltk.FreqDist(token)
        ls_fig.append(create_word_graph(voc))

        ls_group = list(review_df.groupby(['year', 'month']).groups.keys())[::-1]

        review_group = review_df.groupby(['year', 'month'])
        for group in ls_group:
            review = ' '.join(review_group.get_group(group)[['date', 'user_review']].user_review).lower()

            token = tokenizer.tokenize(review)
            if len(keywords) == 0:
                token = [lemmatizer.lemmatize(lemmatizer.lemmatize(word, pos='v'), pos='n') for word in token if
                         word.lower() not in stopwords]
            else:
                token = [lemmatizer.lemmatize(lemmatizer.lemmatize(word, pos='v'), pos='n') for word in token if
                         lemmatizer.lemmatize(lemmatizer.lemmatize(word, pos='v'), pos='n') in keywords]
            voc = nltk.FreqDist(token)

            df = review_group.get_group(group)[['date', 'user_review']]
            df.user_review = df.user_review.map(lambda x: sent_tokenize(x.lower()))
            ls = df.user_review.tolist()

            new_ls = []
            for i in ls:
                new_ls += i
            key_set = set(voc.keys())
            dic = {}
            new_punc = punc.replace("'", "")
            for sent in new_ls:
                wrds_ls = sent.split()
                sumi = 0
                for wrd in wrds_ls:

                    m = lemmatizer.lemmatize(
                        lemmatizer.lemmatize(wrd.translate(str.maketrans("", "", new_punc)), pos='v'), pos='n')
                    if m in key_set:
                        sumi += voc[m]
                dic[sent] = sumi
            if len(dic.values()) == 0:
                continue
            high_freq = max(dic.values())
            if high_freq == 0:
                continue
            df_sent = pd.DataFrame([dic.keys(), dic.values()]).T.rename(columns={0: "sent", 1: "freq"})
            df_sent['weight_freq'] = df_sent.freq.map(lambda x: round(x / high_freq, 2))
            df_sent = df_sent.sort_values(['weight_freq'], ascending=False)
            dic_review['Period : ' + str(group[0]) + '-' + str(group[1])] = df_sent[df_sent.weight_freq > 0.13]
        ls_dic_review.append(dic_review)

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
        height=165, width=500,
        xaxis=dict(showticklabels=False, ticks="", fixedrange=True, ),
        yaxis={**axis_style, **{'tick0': 0, 'dtick': 1}},
        margin=margin_style,
        plot_bgcolor='white',
    )

    overall_df = my_reviews_df  # TODO: (nice to have) get rid of this overall_df if it is really just a copy of my_reviews_df

    overall_df.dropna(subset=["user_review"], inplace=True)
    overall_review = ' '.join(overall_df[['date', 'user_review']].user_review).lower()
    # overall_df['date'] = overall_df.date.map(lambda x: x.date())

    overall_df['user_reviews'] = overall_df['user_review'].map(
        lambda x: x.lower().translate(str.maketrans("", "", new_punc)))
    tokenizer = RegexpTokenizer("[\w']+")
    overall_token = tokenizer.tokenize(' '.join(overall_df.user_reviews))
    overall_token = [
        lemmatizer.lemmatize(lemmatizer.lemmatize(word.translate(str.maketrans("", "", new_punc)), pos='v')) for word in
        overall_token if word.lower() not in stopwords]
    overall_voc = nltk.FreqDist(overall_token)

    overall_df['user_reviews'] = overall_df.user_review.map(lambda x: sent_tokenize(x.lower()))
    ls = overall_df.user_reviews.tolist()

    new_ls = []
    for i in ls:
        new_ls += i
    key_set = set(overall_voc.keys())
    dic = {}
    new_punc = punc.replace("'", "")
    for sent in new_ls:
        wrds_ls = sent.split()
        sumi = 0
        for wrd in wrds_ls:
            m = lemmatizer.lemmatize(lemmatizer.lemmatize(wrd.translate(str.maketrans("", "", new_punc)), pos='v'),
                                     pos='n')
            if m in key_set:
                sumi += overall_voc[m]
        dic[sent] = sumi
    high_freq = max(dic.values())
    df_sent = pd.DataFrame([dic.keys(), dic.values()]).T.rename(columns={0: "sent", 1: "freq"})
    df_sent['weight_freq'] = df_sent.freq.map(lambda x: round(x / high_freq, 2))
    df_sent = df_sent.sort_values(['weight_freq'], ascending=False)

    # from monkeylearn import MonkeyLearn

    text = ' '.join(df_sent[df_sent.weight_freq > 0.13].sent)

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
        height=400,
        title_text="Overview of Review Rating",
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


    def generate_reviews_st(group, dic_review):
        text = '-----'.join(dic_review[group].sent)
        text = group + ":  " + text
        return text


    ls_options_1_st = ["last {} months".format(i + 1) for i in
                       range(1, len(ls_dic_review[0].keys()))]
    ls_options_2_st = ["last {} months".format(i + 1) for i in
                       range(1, len(ls_dic_review[1].keys()))]
    ls_options_3_st = ["last {} months".format(i + 1) for i in
                       range(1, len(ls_dic_review[2].keys()))]
    ls_options_4_st = ["last {} months".format(i + 1) for i in
                       range(1, len(ls_dic_review[3].keys()))]
    ls_options_5_st = ["last {} months".format(i + 1) for i in
                       range(1, len(ls_dic_review[4].keys()))]
    text_center = {'text-align': 'center'}
    component_width = {'min-width': '1000px', 'margin': 'auto'}


    option = st.selectbox(
         'Which ratings do you want to analyze?',
         ('Reviews Rating 1', 'Reviews Rating 2', 'Reviews Rating 3','Reviews Rating 4','Reviews Rating 5'))

    if option == 'Reviews Rating 1':
        st.subheader("Reviews Rating 1")
        st.plotly_chart(ls_fig[0])

        selected_month = st.selectbox(
            'Select months',
            ls_options_1_st
        )
        number_months = int(re.findall('\d*\.?\d+', selected_month)[0])
        text_to_show = [generate_reviews_st(group, ls_dic_review[0]) for group in
                        list(ls_dic_review[0].keys())[:number_months]]
        st.write(text_to_show)
    elif option == 'Reviews Rating 2':
        st.subheader("Reviews Rating 2")
        st.plotly_chart(ls_fig[1])

        selected_month = st.selectbox(
            'Select months',
            ls_options_2_st
        )
        number_months = int(re.findall('\d*\.?\d+', selected_month)[0])
        text_to_show = [generate_reviews_st(group, ls_dic_review[1]) for group in
                        list(ls_dic_review[1].keys())[:number_months]]
        st.write(text_to_show)
    elif option == 'Reviews Rating 3':
        st.subheader("Reviews Rating 3")
        st.plotly_chart(ls_fig[2])

        selected_month = st.selectbox(
            'Select months',
            ls_options_3_st
        )
        number_months = int(re.findall('\d*\.?\d+', selected_month)[0])
        text_to_show = [generate_reviews_st(group, ls_dic_review[2]) for group in
                        list(ls_dic_review[2].keys())[:number_months]]
        st.write(text_to_show)
    elif option == 'Reviews Rating 4':
        st.subheader("Reviews Rating 4")
        st.plotly_chart(ls_fig[3])

        selected_month = st.selectbox(
            'Select months',
            ls_options_4_st
        )
        number_months = int(re.findall('\d*\.?\d+', selected_month)[0])
        text_to_show = [generate_reviews_st(group, ls_dic_review[3]) for group in
                        list(ls_dic_review[3].keys())[:number_months]]
        st.write(text_to_show)
    elif option == 'Reviews Rating 5':
        st.subheader("Reviews Rating 5")
        st.plotly_chart(ls_fig[4])

        selected_month = st.selectbox(
            'Select months',
            ls_options_5_st
        )
        number_months = int(re.findall('\d*\.?\d+', selected_month)[0])
        text_to_show = [generate_reviews_st(group, ls_dic_review[4]) for group in
                        list(ls_dic_review[4].keys())[:number_months]]
        st.write(text_to_show)
    else:
        pass 
    #option == 'Reviews Rating All':
    #st.subheader("Reviews Rating All")
    #st.plotly_chart(fig_overall)