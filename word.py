import streamlit as st

libs = [
    "feedparser",
    "pandas",
    "sklearn",
    "wordcloud",
    "matplotlib",
    "vaderSentiment"
]

for lib in libs:
    try:
        __import__(lib)
        st.success(f"{lib} loaded")
    except ImportError:
        st.error(f"{lib} NOT installed")


import streamlit as st
import feedparser
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
from collections import Counter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ---------------------------------
# PAGE CONFIG
# ---------------------------------
st.set_page_config(
    page_title="Advanced Trending Topics Dashboard",
    layout="wide",
    page_icon="🔥"
)

# ---------------------------------
# CUSTOM UI STYLE
# ---------------------------------
st.markdown("""
<style>
.metric {
    background-color: #1f1f1f;
    padding: 18px;
    border-radius: 14px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------
# TITLE
# ---------------------------------
st.title("🔥 Advanced Trending Topics Dashboard")
st.caption("Search · Trends · Sentiment · Country Analysis · Massive Word Clouds")

# ---------------------------------
# SIDEBAR CONTROLS
# ---------------------------------
st.sidebar.header("⚙️ Dashboard Controls")

search_word = st.sidebar.text_input("🔍 Search keyword (global)", "")
max_words = st.sidebar.number_input(
    "☁️ Word Cloud Max Words",
    min_value=50,
    max_value=2000,
    value=1000,
    step=50
)

country = st.sidebar.selectbox(
    "🌐 Select Country",
    {
        "Global": "",
        "India": "&hl=en-IN&gl=IN",
        "USA": "&hl=en-US&gl=US",
        "UK": "&hl=en-GB&gl=GB",
        "Australia": "&hl=en-AU&gl=AU"
    }
)

show_raw = st.sidebar.checkbox("Show raw titles", False)

# ---------------------------------
# RSS SOURCES
# ---------------------------------
SOURCES = {
    "Twitter": "https://news.google.com/rss/search?q=trending+on+twitter",
    "Facebook": "https://news.google.com/rss/search?q=trending+on+facebook",
    "Reddit": "https://www.reddit.com/r/popular/.rss"
}

# ---------------------------------
# FETCH DATA
# ---------------------------------
def fetch_feed(url, source):
    feed = feedparser.parse(url + country)
    data = []

    for entry in feed.entries:
        data.append({
            "title": entry.title,
            "published": entry.get("published", "N/A"),
            "source": source
        })

    return pd.DataFrame(data)

# ---------------------------------
# LOAD ALL DATA
# ---------------------------------
dfs = [fetch_feed(url, name) for name, url in SOURCES.items()]
df_all = pd.concat(dfs, ignore_index=True)

# ---------------------------------
# SEARCH FILTER
# ---------------------------------
if search_word:
    df_all = df_all[df_all["title"].str.contains(search_word, case=False)]

# ---------------------------------
# METRICS
# ---------------------------------
col1, col2, col3 = st.columns(3)
col1.metric("Total Records", len(df_all))
col2.metric("Platforms", df_all["source"].nunique())
col3.metric("Country Mode", "Global" if country == "" else country)

# ---------------------------------
# RAW DATA
# ---------------------------------
if show_raw:
    with st.expander("📝 Raw Feed Data"):
        st.dataframe(df_all)

# ---------------------------------
# TF-IDF
# ---------------------------------
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df_all["title"])
scores = tfidf_matrix.sum(axis=0).A1
words = vectorizer.get_feature_names_out()

tfidf_df = pd.DataFrame({
    "Word": words,
    "TF-IDF Score": scores
}).sort_values("TF-IDF Score", ascending=False).head(max_words)

# ---------------------------------
# SENTIMENT ANALYSIS
# ---------------------------------
analyzer = SentimentIntensityAnalyzer()

df_all["Sentiment Score"] = df_all["title"].apply(
    lambda x: analyzer.polarity_scores(x)["compound"]
)

df_all["Sentiment"] = df_all["Sentiment Score"].apply(
    lambda x: "Positive" if x > 0.05 else "Negative" if x < -0.05 else "Neutral"
)

with st.expander("🤖 Sentiment Analysis Table"):
    st.dataframe(df_all[["title", "Sentiment", "Sentiment Score"]])

# ---------------------------------
# TIME-BASED TREND (TABLE)
# ---------------------------------
df_all["published"] = pd.to_datetime(df_all["published"], errors="coerce")
trend_table = df_all.groupby(df_all["published"].dt.date).size().reset_index(name="Count")

with st.expander("📈 Time-based Trend (Table)"):
    st.dataframe(trend_table)

# ---------------------------------
# WORD CLOUD
# ---------------------------------
st.subheader("☁️ Massive Word Cloud")

wc = WordCloud(
    width=1800,
    height=900,
    max_words=max_words,
    background_color="white",
    colormap="magma"
).generate_from_frequencies(dict(zip(tfidf_df["Word"], tfidf_df["TF-IDF Score"])))

fig, ax = plt.subplots(figsize=(18, 9))
ax.imshow(wc)
ax.axis("off")
st.pyplot(fig)

# ---------------------------------
# KEYWORD CONTEXT
# ---------------------------------
st.subheader("🎯 Analyze Specific Word")

selected_word = st.text_input("Enter word to analyze")

if selected_word:
    filtered = df_all[df_all["title"].str.contains(selected_word, case=False)]

    st.metric("Occurrences", len(filtered))

    with st.expander("📄 Titles Containing Word"):
        st.dataframe(filtered[["title", "source", "Sentiment"]])

# ---------------------------------
# FOOTER
# ---------------------------------
st.markdown("---")
st.markdown("🚀 **Enterprise-Grade NLP Dashboard | Streamlit | TF-IDF | Sentiment | RSS Intelligence**")
