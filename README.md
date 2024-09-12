# Amazon Reviews Data Analysis Documentation

This document provides a detailed explanation of a Python script that performs data analysis on Amazon reviews. The script covers various aspects of data manipulation, visualization, and sentiment analysis.

## Table of Contents
1. [Library Imports](#1-library-imports)
2. [Data Loading](#2-data-loading)
3. [Data Cleaning](#3-data-cleaning)
4. [Data Analysis](#4-data-analysis)
5. [Product Analysis](#5-product-analysis)
6. [User Behavior Analysis](#6-user-behavior-analysis)
7. [Text Length Analysis](#7-text-length-analysis)
8. [Sentiment Analysis](#8-sentiment-analysis)
9. [Results](#9-results)

## 1. Library Imports

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from textblob import TextBlob
import warnings
from warnings import filterwarnings
import sqlite3
```

This section imports necessary libraries for data manipulation (pandas, numpy), visualization (matplotlib, seaborn), text analysis (TextBlob), and database operations (sqlite3).

## 2. Data Loading

```python
con = sqlite3.connect(r'N:\Personal_Projects\Machine_learning_projects\amazon_analysis\database.sqlite')
df = pd.read_sql_query("SELECT * FROM REVIEWS", con)
```

Data is loaded from a SQLite database into a pandas DataFrame.

## 3. Data Cleaning

```python
df_valid = df[df['HelpfulnessNumerator'] <= df['HelpfulnessDenominator']]
df_valid = df_valid.drop_duplicates(['UserId', 'ProfileName', 'Time', 'Text'])
```

This section removes invalid entries where HelpfulnessNumerator is greater than HelpfulnessDenominator, and drops duplicate entries based on user ID, profile name, time, and text.

## 4. Data Analysis

```python
recommend_df = df.groupby(['UserId']).agg({
    'Summary': 'count', 
    'Text': 'count', 
    'Score': 'mean', 
    'ProductId': 'count'
}).sort_values(by='ProductId', ascending=True)
```

This creates a summary DataFrame grouping by user ID, showing counts of summaries, texts, mean scores, and number of products reviewed.

## 5. Product Analysis

```python
prod_count = df['ProductId'].value_counts().to_frame()
freq_product_id = prod_count[prod_count['count']>500].index
freq_prod_df = df[df['ProductId'].isin(freq_product_id)]

sns.countplot(y = 'ProductId', data = freq_prod_df, hue='Score')
plt.show()
```

This section identifies frequently reviewed products (more than 500 reviews) and visualizes their score distribution.

## 6. User Behavior Analysis

```python
df['viewerdf'] = df['UserId'].apply(lambda user: "Frequent" if X[user]>50 else "Not Frequent")
not_freq_df = df[df['viewerdf']=='Not Frequent']
freq_df = df[df['viewerdf']=='Frequent']

freq_df['Score'].value_counts()/len(freq_df)*100
not_freq_df['Score'].value_counts()/len(not_freq_df)*100
```

This part categorizes users as frequent (>50 reviews) or not frequent, and analyzes their scoring patterns.

## 7. Text Length Analysis

```python
def calc_len(text):
    return len(text.split(' '))
df['Text_len'] = df['Text'].apply(calc_len)

fig = plt.figure(figsize=(10,5))
ax1 = fig.add_subplot(121)
ax1.boxplot(not_freq_df['Text_len'])
ax2 = fig.add_subplot(122)
ax2.boxplot(freq_df['Text_len'])
```

This section calculates the length of review texts and compares the distribution between frequent and non-frequent users.

## 8. Sentiment Analysis

```python
sample = df[0:50000]
polarity = []
for text in sample['Summary']:
    try:
        polarity.append(TextBlob(text).sentiment.polarity)
    except:
        polarity.append(0)
sample['polarity'] = polarity

sample_negative = sample[sample['polarity']<0]
sample_positive = sample[sample['polarity']>0]

from collections import Counter
Counter(sample_negative['Summary']).most_common(10)
Counter(sample_positive['Summary']).most_common(10)
```

This part performs sentiment analysis on a sample of 50,000 reviews, categorizing them as positive or negative based on polarity, and identifies the most common words in each category.

## 9. Results

The key findings from this analysis include:

1. Data Cleaning: [Number of rows removed due to invalid helpfulness scores and duplicates]
2. User Behavior:
   - Number of frequent users (>50 reviews): [To be filled after running]
   - Comparison of scoring patterns between frequent and non-frequent users
3. Product Analysis:
   - Number of products with more than 500 reviews: [To be filled after running]
   - Distribution of scores for frequently reviewed products
4. Text Length Analysis:
   - Comparison of review length between frequent and non-frequent users
5. Sentiment Analysis:
   - Distribution of positive and negative sentiments in the sample
   - Most common words in positive and negative reviews

[Additional insights and visualizations to be added after running the full analysis]

This comprehensive analysis provides insights into user behavior, product reception, and sentiment patterns in Amazon reviews, which can be valuable for understanding customer preferences and improving product offerings.
