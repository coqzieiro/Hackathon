import cohere
import csv
import streamlit as st
from cohere.responses.classify import Example
import pandas as pd
from collections import Counter
import ast
import datetime
from PIL import Image

co = cohere.Client(os.environ.get('COHERE_API_KEY'))

icon = Image.open('icon.png')

st.set_page_config(page_title='SoundFinance', page_icon=icon, layout='wide')
st.title('SoundFinance')

col1, col2 = st.columns(2)

price = csv.DictReader(open('precos.csv', 'r'))

examples=[]
days={}

def most_frequent_element(arr):
    counter = Counter(arr)
    most_common = counter.most_common(1)
    return most_common[0][0]


with open('spoti.csv', 'r') as datafile:
    dataset = csv.DictReader(datafile)
    for row in dataset:
        if row['added'] == '':
            continue
        if row['added'] in days:
            days[row['added']][0] = (int(row["val"]) + days[row['added']][0]) / 2 # get mean of 
            days[row['added']][1].append(row['top genre'])
        else:
            days[row['added']] = [int(row["val"]), [row['top genre']]]

 
    prices = pd.read_csv('precos.csv')
    i = 0
    for day, mood in days.items():
        if i > 2499:
            break

     feel = "negative"
        if int(mood[0]) > 40: #and percent_change > 0:
            feel = "positive"
        
        examples.append(Example(day + ' ' + most_frequent_element(mood[1]), feel))
        i += 1


charts = pd.read_csv('charts2.csv')
charts = charts[charts["country"] == 'us']

#music = st.text_area(':red[Dates] :date:', placeholder='2020-11-19')
date = col1.date_input("Day for sentiment analysis", min_value=datetime.date(2020, 1, 1), max_value=datetime.date(2025, 1, 1))
if col1.button('Analyze'):

    songs = []
    genre = ''
    cts = charts[charts["date"] == str(date)]
    if not cts.empty:
        #st.write(f"{date} not found")
       
        freq = []
        for genres in cts['artist_genres']:
            for genre in ast.literal_eval(genres):
                freq.append(genre)
        genre = most_frequent_element(freq)

    songs.append(str(date) + " " + genre) 
    #print(songs)
        
    response = co.classify(
      model="large",
      inputs=songs,
      examples=examples,
    )
    
    i = 0
    for cl in response.classifications:
        #st.write(f"{cl.prediction}, {round(cl.confidence)*100}%")
        v = -1
        if cl.prediction == "positive":
            v = 1

        response = co.generate(
            model='command',
            prompt=f'If the following line is -1 the average mood is negative, if it is 1 the average mood is positive:\nIf the following line is -1 sell, if it is 1 buy:\n{v}',
            max_tokens=100,
            temperature=0.7,
            k=0,
            stop_sequences=[],
            return_likelihoods='NONE')

        col1.markdown(f"### REPORT {str(date)}:\n{response.generations[0].text}")
        #st.write(f"{dates[i]} | {v}")
        i += 1



with col2:
    st.subheader('SPOT (NYSE)')
    prices = pd.read_csv('precos.csv')

    prices = prices.rename(columns={'Adj Close': 'Price'})
    st.line_chart(prices, x='Date', y='Price', ) #displays the table of data

#st.subheader('ETF S&P 500')
#prices = pd.read_csv('precos2.csv')
#
#st.line_chart(prices, x='Date', y='Adj Close') #displays the table of data

    moods = {} 
    for day, mood in days.items():
        moods[day] = mood[0] / 100.0

    st.subheader('Average mood')
    st.bar_chart(moods) #displays the table of data


