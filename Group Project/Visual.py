import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import re
import plotly.graph_objects as go

from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.express as px

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

import gensim
from gensim import corpora
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer





# Initialize the lemmatizer and stemmer
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()





st.set_page_config(page_title="McDonald's Sentiment Dashboard", layout="wide")
st.title("🍔 McDonald's Review Sentiment Dashboard")

@st.cache_data
def load_data():
    return pd.read_excel("Dataset/Pre-Text Processing Dataset.xlsx")

df = load_data()

# Sidebar filters with 'All' option
st.sidebar.header("🔍 Filters")
sentiment_options = ['All'] + sorted(df['Sentiment'].dropna().unique().tolist())
branch_options = ['All'] + sorted(df['Branch'].dropna().unique().tolist())

sentiment_filter = st.sidebar.selectbox("Select Sentiment", sentiment_options)
branch_filter = st.sidebar.selectbox("Select Branch", branch_options)

# Apply filtering based on selection
filtered_df = df.copy()
if sentiment_filter != 'All':
    filtered_df = filtered_df[filtered_df['Sentiment'] == sentiment_filter]
if branch_filter != 'All':
    filtered_df = filtered_df[filtered_df['Branch'] == branch_filter]

# Create a space
st.markdown("<br><br>", unsafe_allow_html=True)





### METRICS - CARDS

# Create the columns for the metrics
col1, col2, col3, col4 = st.columns(4)

# Get sentiment counts from filtered data
sentiment_counts = filtered_df['Sentiment'].value_counts()

# Calculate the total sentiment count
total_sentiments = sentiment_counts.sum()

# Add a box with custom style
col1.markdown(
    f"""
    <div style="background-color:#2a2a2a; padding: 20px; border-radius: 10px; text-align:center;">
        <p style="font-size: 28px; font-weight: bold; color: white;">Total</p>
        <p style="font-size: 52px; font-weight: bold; color: white;">{total_sentiments}</p>
    </div>
    """, unsafe_allow_html=True)

col2.markdown(
    f"""
    <div style="background-color:#2a2a2a; padding: 20px; border-radius: 10px; text-align:center;">
        <p style="font-size: 28px; font-weight: bold; color: white;">Negative</p>
        <p style="font-size: 52px; font-weight: bold; color: white;">{sentiment_counts.get('Negative', 0)}</p>
    </div>
    """, unsafe_allow_html=True)

col3.markdown(
    f"""
    <div style="background-color:#2a2a2a; padding: 20px; border-radius: 10px; text-align:center;">
        <p style="font-size: 28px; font-weight: bold; color: white;">Neutral</p>
        <p style="font-size: 52px; font-weight: bold; color: white;">{sentiment_counts.get('Neutral', 0)}</p>
    </div>
    """, unsafe_allow_html=True)

col4.markdown(
    f"""
    <div style="background-color:#2a2a2a; padding: 20px; border-radius: 10px; text-align:center;">
        <p style="font-size: 28px; font-weight: bold; color: white;">Positive</p>
        <p style="font-size: 52px; font-weight: bold; color: white;">{sentiment_counts.get('Positive', 0)}</p>
    </div>
    """, unsafe_allow_html=True)

# Create a space
st.markdown("<br><br>", unsafe_allow_html=True)





### 📊LINE CHART - Sentiment Distribution Over Time by Branch

st.markdown(f"### 📈 Number of Reviews Distributed Each Year by {branch_filter}")

# Update your DataFrame as usual
df_time = filtered_df.copy()

# Mapping for time to approximate year
time_map = {
    "a day ago": "2025", "2 days ago": "2025", "3 days ago": "2025", "4 days ago": "2025", "5 days ago": "2025", "6 days ago": "2025",
    "a week ago": "2025", "2 weeks ago": "2025", "3 weeks ago": "2025", "4 weeks ago": "2025",
    "a month ago": "2025", "2 months ago": "2025", "3 months ago": "2025", "4 months ago": "2025",
    "19 hours ago": "2025", "2 hours ago": "2025",
    "5 months ago": "2024", "6 months ago": "2024", "7 months ago": "2024", "8 months ago": "2024", "9 months ago": "2024",
    "10 months ago": "2024", "11 months ago": "2024", "a year ago": "2024",
    "2 years ago": "2023", "3 years ago": "2022", "4 years ago": "2021", "5 years ago": "2020",
    "6 years ago": "2019", "7 years ago": "2018", "8 years ago": "2017"
}

df_time["Review_Year"] = df_time["Time"].apply(lambda x: time_map.get(str(x).strip().lower(), "Other"))
df_time = df_time[df_time["Review_Year"] != "Other"]  # Remove 'Other'

# Group data by year and sentiment
grouped = df_time.groupby(["Review_Year", "Sentiment"]).size().reset_index(name="Count")
ordered_years = [str(y) for y in range(2017, 2026)]
grouped["Review_Year"] = pd.Categorical(grouped["Review_Year"], categories=ordered_years, ordered=True)
pivot_df = grouped.pivot(index="Review_Year", columns="Sentiment", values="Count").fillna(0)

# Create an interactive Plotly graph
fig = go.Figure()

# Add Negative line if Negative sentiment exists in the filtered data
if 'Negative' in pivot_df:
    fig.add_trace(go.Scatter(x=pivot_df.index, y=pivot_df["Negative"], mode='lines+markers', name='Negative', 
                             line=dict(color='red', width=2), marker=dict(size=12)))

# Add Neutral line if Neutral sentiment exists in the filtered data
if 'Neutral' in pivot_df:
    fig.add_trace(go.Scatter(x=pivot_df.index, y=pivot_df["Neutral"], mode='lines+markers', name='Neutral', 
                             line=dict(color='green', width=2), marker=dict(size=12)))

# Add Positive line if Positive sentiment exists in the filtered data
if 'Positive' in pivot_df:
    fig.add_trace(go.Scatter(x=pivot_df.index, y=pivot_df["Positive"], mode='lines+markers', name='Positive', 
                             line=dict(color='blue', width=2), marker=dict(size=12)))

# Update layout for the graph
fig.update_layout(
    yaxis_title="Number of Reviews",
    plot_bgcolor='#2f2f2f',  # Dark background
    paper_bgcolor='#2f2f2f',  # Dark background for the entire figure
    font=dict(color='white'),
    showlegend=True,
    legend=dict(title="Sentiment", x=0.9, y=1.3, font=dict(size=14)),
    hovermode='x unified',  # Show all values on hover for the x-axis
    xaxis=dict(showgrid=False, tickfont=dict(size=18)),  # Remove gridlines and increase x-axis font size
    yaxis=dict(showgrid=False, tickfont=dict(size=16), title_font=dict(size=16, color='white')),  # Remove gridlines and increase y-axis font size
)

# Render the figure in Streamlit
st.plotly_chart(fig)

# Create a space
st.markdown("<br><br><br>", unsafe_allow_html=True)





# 🧭 PIE CHART AND WORD CLOUD

# Layout for charts
st.markdown("### ☸ Sentiment Distribution & Word Cloud")
chart_col1, chart_col2 = st.columns([1, 1])  # Make both columns take equal space

# Sentiment Distribution Pie Chart
with chart_col1:
    st.markdown(f"#### Sentiment Distribution {branch_filter}")

    # Set custom colors for the pie chart
    custom_colors = ['#B70000', '#00B73A', '#0045B7']  # Red for Negative, Green for Neutral, Blue for Positive

    # Calculate sentiment distribution for the filtered data
    sentiment_counts = filtered_df['Sentiment'].value_counts()

    fig1, ax1 = plt.subplots(figsize=(6, 6))  # You can adjust the figure size here
    wedges, texts, autotexts = ax1.pie(sentiment_counts, labels=None, autopct='%1.1f%%', startangle=90, colors=custom_colors, wedgeprops={'edgecolor': 'white'}, pctdistance=0.85)  # Move percentage labels outside
    
    # Add a legend to the pie chart
    ax1.legend(sentiment_counts.index, loc="upper left", fontsize=8, facecolor='grey')

    # Adjust the size and placement of the percentage labels
    for autotext in autotexts:
        autotext.set_fontsize(10)
        autotext.set_fontweight('bold')
        autotext.set_color('white')
        # Manually adjust the position of the label to move it outside
        x, y = autotext.get_position()
        autotext.set_position((x * 1.35, y * 1.35))  # Increase the multiplier to move it further outside

    # Set background face-color
    ax1.set_facecolor('#2f2f2f')  # Dark background color for the plot area

    # Set the entire figure's background color to dark
    fig1.patch.set_facecolor('#2f2f2f')  # Dark background color for the entire figure

    ax1.axis("equal")  # Equal aspect ratio ensures the pie chart is circular

    st.pyplot(fig1)

# Word Cloud for the filtered sentiment & branch
with chart_col2:
    st.markdown(f"#### Word Cloud {branch_filter}")
    cloud_text = " ".join(filtered_df['Cleaned_Reviews'].astype(str))
    wordcloud = WordCloud(width=600, height=400, background_color='#2f2f2f', colormap='coolwarm').generate(cloud_text)
    
    fig2, ax2 = plt.subplots(figsize=(6, 6))  # Same size for the word cloud
    ax2.imshow(wordcloud, interpolation="bilinear")
    ax2.axis("off")

    # Set the entire figure's background color to dark
    fig2.patch.set_facecolor('#2f2f2f')  # Dark background color for the entire figure

    st.pyplot(fig2)

# Create a space
st.markdown("<br><br><br>", unsafe_allow_html=True)





# 📶 BAR CHART (REVIEWS NEGATIVE AND POSITIVE)

# Complaint Word Charts: Negative vs Positive
st.markdown("### 📶 Top 10 Complaint Words: Negative vs Positive Reviews")

# Function to get the top words for both Negative and Positive Sentiment
def get_top_words(sentiment_type, color):
    reviews = filtered_df[filtered_df['Sentiment'] == sentiment_type]['Processed_Reviews']
    words = re.findall(r'\b\w+\b', " ".join(reviews).lower())

    stopwords = {
        'the', 'and', 'for', 'with', 'this', 'that', 'was', 'are', 'have', 'but', 'not', 'you', 'your', 
        'wait', 'get', 'mud', 'like', 'thou', 'one', 'give','custom', 'even', 'fire', 
        'ask', 'still', 'worst', 'pleas', 'hour', 'take', 'chicken', 'burgher',
        'macdonald', 'mcd', 'thank'
        }
    
    filtered = [w for w in words if w not in stopwords and len(w) > 2]
    return Counter(filtered).most_common(10), color

# Get negative and positive complaint words
neg_words, neg_color = get_top_words('Negative', "Reds_r")
pos_words, pos_color = get_top_words('Positive', "Blues")

chart_col1, chart_col2 = st.columns(2)

# Negative Chart
with chart_col1:
    if neg_words:
        st.markdown(f"#### Top 10 Complaint Words - {branch_filter}")
        
        words, counts = zip(*neg_words)
        
        fig, ax = plt.subplots(figsize=(6, 6))  # Set figure size
        bars = sns.barplot(x=list(counts), y=list(words), ax=ax, palette=neg_color)
        
        ax.set_xlabel("Frequency", color='white')
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        ax.spines['left'].set_visible(True)
        ax.spines['left'].set_color('white')

        ax.spines['bottom'].set_visible(True)
        ax.spines['bottom'].set_color('white')

        for i, v in enumerate(counts):
            ax.text(v + 0.1, i, str(v), va='center', fontweight='bold', color='white')

        # Set background face-color
        ax.set_facecolor('#2f2f2f')  # Dark background color for the plot area

        # Set the entire figure's background color to dark
        fig.patch.set_facecolor('#2f2f2f')  # Dark background color for the entire figure

        # Change axis label color to white
        ax.tick_params(axis='x', colors='white')  # Change x-axis tick color to white
        ax.tick_params(axis='y', colors='white')  # Change y-axis tick color to white

        st.pyplot(fig)
    else:
        st.info("No negative reviews for this branch.")

    
# Positive Chart
with chart_col2:
    if pos_words:
        st.markdown(f"#### Top 10 Good Words - {branch_filter}")
        
        words, counts = zip(*pos_words)
        
        fig, ax = plt.subplots(figsize=(6, 6))  # Set figure size

        # Apply gradient color from dark to light blue
        cmap = plt.get_cmap("Blues")
        colors = [cmap(1 - i / len(words)) for i in range(len(words))]

        bars = sns.barplot(x=list(counts), y=list(words), ax=ax, palette=colors)
        
        ax.set_xlabel("Frequency", color='white')
        
        for i, v in enumerate(counts):
            ax.text(v + 0.1, i, str(v), va='center', fontweight='bold', color='white')
        
        for spine in ax.spines.values():
            # Hide the top and right spines
            if spine in [ax.spines['top'], ax.spines['right']]:
                spine.set_visible(False)
            else:
                spine.set_visible(True)  # Keep x and y visible
                spine.set_color('white')

        # Set background face-color
        ax.set_facecolor('#2f2f2f')  # Dark background color for the plot area

        # Set the entire figure's background color to dark
        fig.patch.set_facecolor('#2f2f2f')  # Dark background color for the entire figure

        # Change axis label color to white
        ax.tick_params(axis='x', colors='white')  # Change x-axis tick color to white
        ax.tick_params(axis='y', colors='white')  # Change y-axis tick color to white

        st.pyplot(fig)
    else:
        st.info("No positive reviews for this branch.")

# Create a space
st.markdown("<br><br><br>", unsafe_allow_html=True)





# 🔍 SENTIMENT ANALYSIS

# TextBlob function
def get_tb_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0:
        sentiment = "Positive"
    elif polarity < 0:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    return pd.Series([polarity, sentiment], index=["TextBlob Polarity", "TextBlob Sentiment"])

# Initialize VADER analyzer
analyzer = SentimentIntensityAnalyzer()

# VADER function
def get_vader_sentiment(text):
    scores = analyzer.polarity_scores(text)
    compound = scores['compound']
    if compound > 0:
        sentiment = "Positive"
    elif compound < 0:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    return pd.Series([compound, sentiment], index=["VADER Compound", "VADER Sentiment"])

# Apply sentiment functions to filtered data
filtered_df[['TextBlob Polarity', 'TextBlob Sentiment']] = filtered_df['Processed_Reviews'].apply(get_tb_sentiment)
filtered_df[['VADER Compound', 'VADER Sentiment']] = filtered_df['Processed_Reviews'].apply(get_vader_sentiment)

# Display updated dataframe with new sentiment columns
st.markdown("### 📄 Sentiment Analysis")
st.dataframe(filtered_df[['Branch', 'Cleaned_Reviews', 'Sentiment', 'TextBlob Sentiment', 'VADER Sentiment']].reset_index(drop=True))

# Pie-Chart Figure
# Create two columns for side-by-side charts
col1, col2 = st.columns(2)

# TextBlob Sentiment Pie Chart in the first column
with col1:
    textblob_sentiment_counts = filtered_df['TextBlob Sentiment'].value_counts()

    fig_textblob = go.Figure(data=[go.Pie(
        labels=textblob_sentiment_counts.index,
        values=textblob_sentiment_counts,
        hole=0.3,  # To make a donut chart
        textinfo="percent",  # Show percentage
        textposition="outside",  # Move the label outside the chart
        hoverinfo="label+percent",  # Show label and percent on hover
        marker=dict(colors=['#00B73A', '#B70000', '#0045B7'], line=dict(color="black", width=2))  # Colors for Positive, Negative, Neutral
    )])

    fig_textblob.update_layout(
        title="TextBlob Sentiment Distribution",
        title_font=dict(size=20, family="Arial", color="white"),
        font=dict(color='white', size=16, weight="bold"),
    )

    st.plotly_chart(fig_textblob)

# VADER Sentiment Pie Chart in the second column
with col2:
    vader_sentiment_counts = filtered_df['VADER Sentiment'].value_counts()

    fig_vader = go.Figure(data=[go.Pie(
        labels=vader_sentiment_counts.index,
        values=vader_sentiment_counts,
        hole=0.3,  # To make a donut chart
        textinfo="percent",  # Show percentage
        textposition="outside",  # Move the label outside the chart
        hoverinfo="label+percent",  # Show label and percent on hover
        marker=dict(colors=['#00B73A', '#B70000', '#0045B7'], line=dict(color="black", width=2))  # Colors for Positive, Negative, Neutral
    )])

    fig_vader.update_layout(
        title="VADER Sentiment Distribution",
        title_font=dict(size=20, family="Arial", color="white"),
        font=dict(color='white', size=16, weight="bold"),
    )
    st.plotly_chart(fig_vader)

# Create a space
st.markdown("<br><br><br>", unsafe_allow_html=True)





### 🔍 Topic Modeling

# Display the top words and bar chart
st.markdown("### 🗃️ Topic Modeling with LDA")
st.markdown("##### Top 5 words for each topic with Coherence Score")

# Tokenize the cleaned reviews into words
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['Processed_Reviews'])

# Fit the LDA model
n_topics = 4  # Number of topics (you can adjust based on your analysis)
lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
lda.fit(X)

# Get top words for each topic
def get_top_words(model, vectorizer, n_words=5):
    feature_names = vectorizer.get_feature_names_out()
    top_words = {}
    for topic_idx, topic in enumerate(model.components_):
        top_idx = topic.argsort()[-n_words:][::-1]
        top_words[topic_idx] = [(feature_names[i], topic[i]) for i in top_idx]
    return top_words

top_words = get_top_words(lda, vectorizer)

# Create a bar chart using seaborn for each topic
def plot_top_words(top_words, n_topics=4, n_words=5):
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), facecolor='#2f2f2f')  # Set background color to dark

    axes = axes.flatten()  # Flatten axes to make it easier to index

    for topic_idx in range(n_topics):
        words, weights = zip(*top_words[topic_idx])  # Unzip words and weights
        sorted_weights, sorted_words = zip(*sorted(zip(weights, words), reverse=True))  # Sort by weight (descending)

        # Prepare data for seaborn barplot
        data = pd.DataFrame({
            'words': sorted_words,
            'weights': sorted_weights
        })

        # Plot using seaborn's barplot
        sns.barplot(x='weights', y='words', data=data, ax=axes[topic_idx], palette="flare", orient='h')

        # Add weight labels on the bars
        for i, weight in enumerate(sorted_weights):
            axes[topic_idx].text(weight + 0.5, i, f'{weight:.3f}', color='white', fontsize=16, va='center')

        # Customize title and axis font size
        axes[topic_idx].set_title(f"Topic {topic_idx + 1}", fontsize=16, color='white')
        
        # Set x-axis and y-axis labels with font size 14px and white color
        axes[topic_idx].set_xlabel('', fontsize=14, color='white')  # x-axis label
        axes[topic_idx].set_ylabel('', fontsize=14, color='white')  # y-axis label
        
        # Customize ticks
        axes[topic_idx].tick_params(axis='x', labelsize=14, labelcolor='white')  # x-axis ticks
        axes[topic_idx].tick_params(axis='y', labelsize=14, labelcolor='white')  # y-axis ticks

        # Remove gridlines
        axes[topic_idx].grid(False)

        # Set the background color of the axes to dark
        axes[topic_idx].set_facecolor('#2f2f2f')
        axes[topic_idx].spines['top'].set_visible(False)
        axes[topic_idx].spines['right'].set_visible(False)
        axes[topic_idx].spines['left'].set_color('white')
        axes[topic_idx].spines['bottom'].set_color('white')

    # Adjust layout
    plt.tight_layout()
    st.pyplot(fig)

plot_top_words(top_words)
