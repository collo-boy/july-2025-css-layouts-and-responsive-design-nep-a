import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
import streamlit as st
from datetime import datetime

# Set up the page
st.set_page_config(page_title="CORD-19 Analysis", layout="wide")
st.title("📊 CORD-19 Dataset Analysis")
st.markdown("Analyzing COVID-19 research publications metadata")

# Part 1: Data Loading and Basic Exploration
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('metadata.csv', low_memory=False)
        return df
    except:
        st.error("Could not load metadata.csv. Please ensure it's in the correct directory.")
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.stop()

# Display basic info
st.header("📋 Data Overview")
col1, col2, col3 = st.columns(3)
col1.metric("Total Papers", df.shape[0])
col2.metric("Columns", df.shape[1])
col3.metric("Missing Titles", df['title'].isna().sum())

# Part 2: Data Cleaning
st.header("🧹 Data Cleaning")

# Handle missing values
df_clean = df.dropna(subset=['title']).copy()

# Convert dates
df_clean['publish_time'] = pd.to_datetime(df_clean['publish_time'], errors='coerce')
df_clean['year'] = df_clean['publish_time'].dt.year

# Create abstract word count
df_clean['abstract_word_count'] = df_clean['abstract'].apply(
    lambda x: len(str(x).split()) if pd.notnull(x) else 0
)

# Part 3: Analysis and Visualization
st.header("📈 Analysis & Visualizations")

# Yearly publications
yearly_counts = df_clean['year'].value_counts().sort_index()
fig1, ax1 = plt.subplots(figsize=(10, 4))
yearly_counts.plot(kind='bar', ax=ax1, color='skyblue')
ax1.set_title('Publications by Year')
ax1.set_xlabel('Year')
ax1.set_ylabel('Count')
plt.xticks(rotation=45)

# Top journals
top_journals = df_clean['journal'].value_counts().head(10)
fig2, ax2 = plt.subplots(figsize=(10, 4))
top_journals.plot(kind='bar', ax=ax2, color='lightgreen')
ax2.set_title('Top 10 Journals')
ax2.set_xlabel('Journal')
ax2.set_ylabel('Count')
plt.xticks(rotation=45)

# Word cloud
all_titles = ' '.join(df_clean['title'].dropna().astype(str))
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_titles)
fig3, ax3 = plt.subplots(figsize=(10, 4))
ax3.imshow(wordcloud, interpolation='bilinear')
ax3.axis('off')
ax3.set_title('Common Words in Titles')

# Display charts
col1, col2 = st.columns(2)
col1.pyplot(fig1)
col2.pyplot(fig2)
st.pyplot(fig3)

# Part 4: Interactive Exploration
st.header("🔍 Interactive Exploration")

# Filters
col1, col2 = st.columns(2)
year_range = col1.slider("Select Year Range", 
                         min_value=int(df_clean['year'].min()), 
                         max_value=int(df_clean['year'].max()),
                         value=(2020, 2022))

min_words = col2.slider("Minimum Abstract Words", 
                        min_value=0, 
                        max_value=500, 
                        value=50)

# Apply filters
filtered_df = df_clean[
    (df_clean['year'] >= year_range[0]) & 
    (df_clean['year'] <= year_range[1]) &
    (df_clean['abstract_word_count'] >= min_words)
]

# Display filtered results
st.write(f"**{len(filtered_df)}** papers match your criteria")
st.dataframe(filtered_df[['title', 'journal', 'year', 'abstract_word_count']].head(10))

# Download option
csv = filtered_df.to_csv(index=False)
st.download_button(
    label="Download Filtered Data",
    data=csv,
    file_name="filtered_cord19.csv",
    mime="text/csv"
