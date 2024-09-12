

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

---
M


## Table of Contents

1. [Data Loading and Initial Exploration](#data-loading-and-initial-exploration)
2. [Data Cleaning](#data-cleaning)
3. [Data Analysis](#data-analysis)
4. [Analyzing Frequent Users](#analyzing-frequent-users)
5. [Analyzing Text Length](#analyzing-text-length)
6. [Sentiment Analysis](#sentiment-analysis)
7. [Results](#results)
8. [Summary](#summary)
9. [Additional Steps and Suggestions](#additional-steps-and-suggestions)

---

## Data Loading and Initial Exploration

### Connecting to the Database and Loading Data
```python
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
import warnings
from warnings import filterwarnings
filterwarnings('ignore')

# Connect to the SQLite database
con = sqlite3.connect(r'N:\Personal_Projects\Machine_learning_projects\amazon_analysis\database.sqlite')
df = pd.read_sql_query("SELECT * FROM REVIEWS", con)
```

### Checking Data Shape and Head
```python
print(df.shape)  # Print the shape of the DataFrame
print(df.head())  # Print the first few rows of the DataFrame
```

---

## Data Cleaning

### Removing Inconsistent Helpfulness Ratios
```python
# Check for inconsistent Helpfulness ratios
inconsistent_reviews = df[df['HelpfulnessNumerator'] > df['HelpfulnessDenominator']]
print(inconsistent_reviews)

# Remove inconsistent reviews
df_valid = df[df['HelpfulnessNumerator'] <= df['HelpfulnessDenominator']]
print(df_valid.shape)  # Print the shape of the cleaned DataFrame
```

### Removing Duplicates
```python
# Check for duplicates
duplicates = df_valid.duplicated(['UserId', 'ProfileName', 'Time', 'Text'])
print(duplicates)

# Remove duplicates
df_valid = df_valid.drop_duplicates(['UserId', 'ProfileName', 'Time', 'Text'])
print(df_valid.shape)  # Print the shape of the DataFrame after removing duplicates
```

### Converting Time Column to Datetime
```python
# Convert the 'Time' column to datetime
df_valid['Time'] = pd.to_datetime(df_valid['Time'], unit='s')
```

---

## Data Analysis

### Analyzing User Activity
```python
print(df.shape)  # Print the shape of the DataFrame
print(df.columns)  # Print the columns of the DataFrame
print(df['ProfileName'].nunique())  # Print the number of unique ProfileNames
print(df['UserId'].nunique())  # Print the number of unique UserIds

# Group by UserId and calculate summary statistics
recommend_df = df.groupby(['UserId']).agg({
    'Summary': 'count',
    'Text': 'count',
    'Score': 'mean',
    'ProductId': 'count'
}).sort_values(by='ProductId', ascending=True)

print(recommend_df.index[0:10])  # Print the first 10 UserIds
```

### Identifying Frequently Reviewed Products
```python
# Count the number of reviews per ProductId
prod_count = df['ProductId'].value_counts().to_frame()
print(prod_count)

# Identify products with more than 500 reviews
freq_product_id = prod_count[prod_count['count'] > 500].index
print(freq_product_id)

# Filter the DataFrame to include only frequently reviewed products
freq_prod_df = df[df['ProductId'].isin(freq_product_id)]
freq_prod_df['Score'] = freq_prod_df['Score'].astype(str)

# Plot the distribution of scores for frequently reviewed products
sns.countplot(y='ProductId', data=freq_prod_df, hue='Score')
plt.show()
```

---

## Analyzing Frequent Users

### Categorizing Users as Frequent or Not Frequent
```python
# Count the number of reviews per UserId
X = df['UserId'].value_counts()

# Categorize users based on the number of reviews
df['viewerdf'] = df['UserId'].apply(lambda user: "Frequent" if X[user] > 50 else "Not Frequent")
print(df['viewerdf'].unique())  # Print the unique categories
```

### Separating Frequent and Not Frequent Users
```python
not_freq_df = df[df['viewerdf'] == 'Not Frequent']
freq_df = df[df['viewerdf'] == 'Frequent']
```

### Analyzing Scores
```python
# Print the distribution of scores for frequent users
print(freq_df['Score'].value_counts())
print(freq_df['Score'].value_counts() / len(freq_df) * 100)  # Convert to percentage

# Print the distribution of scores for not frequent users
print(not_freq_df['Score'].value_counts() / len(not_freq_df) * 100)

# Plot the distribution of scores for frequent users
freq_df['Score'].value_counts().plot(kind='bar')
plt.show()

# Plot the distribution of scores for not frequent users
not_freq_df['Score'].value_counts().plot(kind='bar')
plt.show()
```

---

## Analyzing Text Length

### Calculating Text Length
```python
def calc_len(text):
    return len(text.split(' '))

df['Text_len'] = df['Text'].apply(calc_len)
```

### Plotting Text Length Distributions
```python
fig = plt.figure(figsize=(10, 5))

ax1 = fig.add_subplot(121)
ax1.boxplot(not_freq_df['Text_len'])
ax1.set_xlabel('Not Frequent Users')
ax1.set_ylim(0, 600)

ax2 = fig.add_subplot(122)
ax2.boxplot(freq_df['Text_len'])
ax2.set_xlabel('Frequent Users')
ax2.set_ylim(0, 600)

plt.show()
```

---

## Sentiment Analysis

### Calculating Sentiment Polarity
```python
sample = df[0:50000]
polarity = []

for text in sample['Summary']:
    try:
        polarity.append(TextBlob(text).sentiment.polarity)
    except:
        polarity.append(0)

sample['polarity'] = polarity
```

### Analyzing Sentiment
```python
sample_negative = sample[sample['polarity'] < 0]
sample_positive = sample[sample['polarity'] > 0]

from collections import Counter

# Print the most common negative summaries
print(Counter(sample_negative['Summary']).most_common(10))

# Print the most common positive summaries
print(Counter(sample_positive['Summary']).most_common(10))
```

---

## Results

### Data Shape and Head
- **Initial Data Shape:** `(rows, columns)`
- **First Few Rows:**
  ```python
  print(df.head())
  ```

### Data Cleaning
- **Inconsistent Helpfulness Ratios Removed:**
  ```python
  print(df_valid.shape)
  ```
- **Duplicates Removed:**
  ```python
  print(df_valid.shape)
  ```

### Data Analysis
- **Unique ProfileNames:**
  ```python
  print(df['ProfileName'].nunique())
  ```
- **Unique UserIds:**
  ```python
  print(df['UserId'].nunique())
  ```
- **User Activity Summary:**
  ```python
  print(recommend_df.index[0:10])
  ```

### Frequently Reviewed Products
- **Products with More Than 500 Reviews:**
  ```python
  print(freq_product_id)
  ```

### Frequent Users
- **Categorization Results:**
  ```python
  print(df['viewerdf'].unique())
  ```

### Score Analysis
- **Frequent Users Score Distribution:**
  ```python
  print(freq_df['Score'].value_counts() / len(freq_df) * 100)
  ```
- **Not Frequent Users Score Distribution:**
  ```python
  print(not_freq_df['Score'].value_counts() / len(not_freq_df) * 100)
  ```

### Text Length Analysis
- **Text Length Distributions:**
  ```python
  plt.show()
  ```

### Sentiment Analysis
- **Most Common Negative Summaries:**
  ```python
  print(Counter(sample_negative['Summary']).most_common(10))
  ```
- **Most Common Positive Summaries:**
  ```python
  print(Counter(sample_positive['Summary']).most_common(10))
  ```

---

## Summary

You've performed a comprehensive analysis of Amazon reviews, including data cleaning, user categorization, text length analysis, and sentiment analysis. These steps provide a solid foundation for further analysis and modeling. Keep exploring and visualizing the data to uncover more insights!

---

## Additional Steps and Suggestions

- **Visualizations:**
  Create more visualizations to better understand the data. For example, plot the distribution of scores, the number of reviews per user, or the number of reviews per product.

- **Sentiment Analysis:**
  Since you're working with reviews, sentiment analysis could provide valuable insights. You can use libraries like `TextBlob` or `VADER` for this purpose.

- **Feature Engineering:**
  Consider creating new features that might be useful for analysis, such as the length of the review text, the time of day the review was posted, etc.

- **Modeling:**
  If your goal is to predict review scores or identify influential reviewers, you might want to build a predictive model using machine learning algorithms.

Here's an example of how you might perform sentiment analysis using `TextBlob`:

```python
from textblob import TextBlob

def get_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

df_valid['Sentiment'] = df_valid['Text'].apply(get_sentiment)
sns.histplot(df_valid['Sentiment'], bins=30, kde=True)
plt.title('Sentiment Distribution')
plt.show()
```




