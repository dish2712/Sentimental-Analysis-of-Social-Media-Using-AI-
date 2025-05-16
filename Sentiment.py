# ======================== IMPORTS ========================
import streamlit as st
import base64
import re
import pickle
import pandas as pd
import plotly.express as px
from textblob import TextBlob
from streamlit_lottie import st_lottie
import requests

# ======================== LOTTIE SETUP ========================
def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_map = {
    "Amazon Alexa Review Analysis": load_lottie_url("https://assets4.lottiefiles.com/packages/lf20_HpFqiS.json"),
    "Social Media Sentiment Analysis": load_lottie_url("https://assets10.lottiefiles.com/packages/lf20_1pxqjqps.json"),
    "Live News Sentiment Analysis": load_lottie_url("https://assets2.lottiefiles.com/packages/lf20_w51pcehl.json")
}

# ======================== BACKGROUND IMAGE ========================
def add_bg_from_local(image_file):
    with open(image_file, "rb") as file:
        encoded_string = base64.b64encode(file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_string}");
            background-size: cover;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

def set_background_by_page(page_name):
    image_map = {
        "Amazon Alexa Review Analysis": "2.avif",
        "Social Media Sentiment Analysis": "face.png",
         "Live News Sentiment Analysis": "news.avif"
    }
    selected_img = image_map.get(page_name, "1.webp")
    add_bg_from_local(selected_img)

# ======================== HELPER FUNCTIONS ========================
# ======================== HELPER FUNCTIONS ========================
def clean_and_split_input(text):
    sentences = text.strip().split('\n')
    return [s.strip() for s in sentences if len(s.strip()) > 0]

def get_emoji(sentiment):
    return "😃" if sentiment == "POSITIVE" else ("😞" if sentiment == "NEGATIVE" else "😐")

def get_textblob_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.1:
        return "POSITIVE", "😃"
    elif polarity < -0.1:
        return "NEGATIVE", "😞"
    else:
        return "NEUTRAL", "😐"

import plotly.graph_objects as go

def display_sentiment_pie(sentiment_counts, title="Sentiment Distribution"):
    sentiment_df = sentiment_counts.reset_index()
    sentiment_df.columns = ['Sentiment', 'Count']
    emoji_map = {"POSITIVE": "😃", "NEGATIVE": "😞", "NEUTRAL": "😐"}
    sentiment_df['Sentiment'] = sentiment_df['Sentiment'].map(lambda x: f"{emoji_map.get(x, '')} {x}")

    fig = px.pie(
        sentiment_df,
        names='Sentiment',
        values='Count',
        title=title,
        hole=0.4,
        color='Sentiment',
        color_discrete_map={
            "😃 POSITIVE": "green",
            "😞 NEGATIVE": "red",
            "😐 NEUTRAL": "orange"
        }
    )
    fig.update_traces(textinfo='label+percent', insidetextorientation='radial')
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title_font=dict(size=20),
    )
    st.plotly_chart(fig, use_container_width=True)

# ======================== LOAD MODEL AND VECTORIZER ========================
with open('model.pickle', 'rb') as f:
    model = pickle.load(f)
with open('vector.pickle', 'rb') as f:
    vectorizer = pickle.load(f)

# ======================== SIDEBAR NAVIGATION ========================
with st.sidebar:
    st.markdown("<h2 style='text-align:center;'>📊 Choose Analysis</h2>", unsafe_allow_html=True)
    page = st.radio("", options=[
        "Amazon Alexa Review Analysis",
        "Social Media Sentiment Analysis",
        "Live News Sentiment Analysis"
    ], format_func=lambda x: f"🔹 {x}")

# ======================== PAGE BACKGROUND ========================
set_background_by_page(page)

# ======================== AMAZON ALEXA REVIEW PAGE ========================
if page == "Amazon Alexa Review Analysis":
    st_lottie(lottie_map[page], height=200, key="alexa")
    st.markdown("<h1 style='text-align:center;'>Amazon Alexa Review Analysis</h1><hr>", unsafe_allow_html=True)

    user_input = st.text_area("📝 Enter multiple reviews (one per line)", height=250)

    if st.button("🔍 Predict Sentiment"):
        sentences = clean_and_split_input(user_input)
        if not sentences:
            st.warning("⚠️ Please enter at least one review.")
        else:
            transformed = vectorizer.transform(sentences)
            predictions = model.predict(transformed)

            sentiments = ["POSITIVE" if p == 1 else "NEGATIVE" for p in predictions]
            emojis = [get_emoji(s) for s in sentiments]

            df = pd.DataFrame({
                "Review": sentences,
                "Sentiment": sentiments,
                "Emoji": emojis
            })

            count = df['Sentiment'].value_counts()

            st.markdown(f"<h4 style='color:#6a0dad;'>📋 Total Reviews: {len(sentences)}</h4>", unsafe_allow_html=True)
            st.dataframe(df)

            st.subheader("🧾 Emoji Summary")
            for _, row in df.iterrows():
                st.markdown(f"""
                    <div style='background: #ffffffdd; padding: 10px; border-radius: 10px; margin-bottom: 8px; box-shadow: 2px 2px 8px #88888830;'>
                        <b>{row['Emoji']} {row['Review']}</b>
                    </div>
                """, unsafe_allow_html=True)

            st.subheader("🥧 Sentiment Distribution (Pie Chart)")
            display_sentiment_pie(count)

# ======================== SOCIAL MEDIA SENTIMENT PAGE ========================
elif page == "Social Media Sentiment Analysis":
    st_lottie(lottie_map[page], height=200, key="social")
    st.markdown("<h1 style='text-align:center;'>Social Media Sentiment Analysis</h1><hr>", unsafe_allow_html=True)

    sm_input = st.text_area("💬 Paste social media comments (one per line)", height=250)

    if st.button("🔍 Analyze Comments"):
        comments = clean_and_split_input(sm_input)
        if not comments:
            st.warning("⚠️ Please enter at least one comment.")
        else:
            sentiments = []
            emojis = []
            for text in comments:
                sent, emoji = get_textblob_sentiment(text)
                sentiments.append(sent)
                emojis.append(emoji)

            df = pd.DataFrame({
                "Comment": comments,
                "Sentiment": sentiments,
                "Emoji": emojis
            })

            count = df['Sentiment'].value_counts()

            st.markdown(f"<h4 style='color:#6a0dad;'>📋 Total Comments: {len(comments)}</h4>", unsafe_allow_html=True)
            st.dataframe(df)

            st.subheader("🧾 Emoji Summary")
            for _, row in df.iterrows():
                st.markdown(f"""
                    <div style='background: #ffffffdd; padding: 10px; border-radius: 10px; margin-bottom: 8px; box-shadow: 2px 2px 8px #88888830;'>
                        <b>{row['Emoji']} {row['Comment']}</b>
                    </div>
                """, unsafe_allow_html=True)

            st.subheader("📊 Sentiment Distribution")
            display_sentiment_pie(count)

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download as CSV", data=csv, file_name="social_sentiment.csv", mime="text/csv")

elif page == "Live News Sentiment Analysis":
    st_lottie(lottie_map[page], height=200, key="news")
    st.markdown("<h1 style='text-align:center;'>Live News Sentiment Analysis</h1><hr>", unsafe_allow_html=True)

    with st.expander("🌍 Live News Sentiment Feed (via NewsData.io)"):
        topic = st.text_input("🔍 Enter a topic (e.g. AI, Mental Health, Politics):", value="mental health")
        sentiment_filter = st.selectbox("Sentiment Filter", ["positive", "negative", "neutral", "all"])
        max_articles = st.slider("Max articles", 5, 50, 10)

        if st.button("🚀 Fetch & Analyze News"):
            base_url = "https://newsdata.io/api/1/latest"
            params = {
                "apikey": "pub_82988a7355948a3fba0cb1c4c55f834c4690a",
                "q": topic,
                "language": "en",
            }
            if sentiment_filter != "all":
                params["sentiment"] = sentiment_filter

            with st.spinner("Fetching news articles..."):
                response = requests.get(base_url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    articles = data.get("results", [])[:max_articles]
                    if articles:
                        texts = [article['title'] for article in articles if article.get('title')]

                        sentiments = []
                        emojis = []
                        for text in texts:
                            sent, emoji = get_textblob_sentiment(text)
                            sentiments.append(sent)
                            emojis.append(emoji)

                        df_news = pd.DataFrame({
                            "Title": texts,
                            "Sentiment": sentiments,
                            "Emoji": emojis
                        })

                        count = df_news['Sentiment'].value_counts()

                        st.success(f"Fetched {len(texts)} news headlines for '{topic}'")
                        st.dataframe(df_news)

                        st.subheader("🧾 Headline Summary")
                        for i, row in df_news.iterrows():
                            st.markdown(f"""
                                <div style='background: #ffffffdd; padding: 10px; border-radius: 10px; margin-bottom: 8px; box-shadow: 2px 2px 8px #88888830;'>
                                    <b>{row['Emoji']} {row['Title']}</b>
                                </div>
                            """, unsafe_allow_html=True)

                        st.subheader("🥧 Sentiment Distribution")
                        display_sentiment_pie(count)

                        csv = df_news.to_csv(index=False).encode('utf-8')
                        st.download_button("📥 Download Results", data=csv, file_name="news_sentiment.csv", mime="text/csv")
                    else:
                        st.warning("No articles found.")
                else:
                    st.error("Failed to fetch news. Check your API key or try again later.")