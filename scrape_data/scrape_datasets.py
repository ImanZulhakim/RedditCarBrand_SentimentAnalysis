import math
from pprint import pprint
import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import praw
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
from IPython.display import display, clear_output

sns.set(style='darkgrid', context='talk', palette='Dark2')

reddit = praw.Reddit(client_id='UwQGu_BZV3plC9jLCXaRTg',
                     client_secret='pIR4KDYFE0cxjBh73acT7voeSEKm7g',
                     user_agent='LapSent')

# Initialize lists for headlines, dates, URLs, and labels
headlines = []
dates = []
urls = []
labels = []

# Fetch data from subreddit reddit.subreddit('subreddit_name')
for submission in reddit.subreddit('mazda').new(limit=None):
    headlines.append(submission.title)
    dates.append(submission.created_utc)
    urls.append(submission.url)
    clear_output()
    print(len(headlines))

# Sentiment analysis using NLTK's Vader
sia = SIA()
results = []

for line in headlines:
    pol_score = sia.polarity_scores(line)
    pol_score['headline'] = line
    results.append(pol_score)

pprint(results[:30], width=100)

# Construct DataFrame from sentiment analysis results
df = pd.DataFrame.from_Srecords(results)

# Add labels based on compound scores
df['label'] = 0
df.loc[df['compound'] > 0.2, 'label'] = 1
df.loc[df['compound'] < -0.2, 'label'] = -1

# Add date, URL, and label to the DataFrame
df['date'] = dates
df['url'] = urls

# Convert Unix timestamp to human-readable date and time
df['date'] = pd.to_datetime(df['date'], unit='s')

# Reorder columns
df2 = df[['headline', 'date', 'url']]

# Save DataFrame to CSV
df2.to_csv('mazda.csv', index=False)

print("Positive headlines:\n")
pprint(list(df[df['label'] == 1].headline)[:5], width=200)

print("\nNegative headlines:\n")
pprint(list(df[df['label'] == -1].headline)[:5], width=200)

print(df.label.value_counts())

print(df.label.value_counts(normalize=True) * 100)

fig, ax = plt.subplots(figsize=(8, 8))

counts = df.label.value_counts(normalize=True) * 100

sns.barplot(x=counts.index, y=counts, ax=ax)

ax.set_xticklabels(['Negative', 'Neutral', 'Positive'])
ax.set_ylabel("Percentage")

plt.show()

from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords

stop_words = stopwords.words('english')
tokenizer = RegexpTokenizer(r'\w+')


def process_text(headlines):
    tokens = []
    for line in headlines:
        toks = tokenizer.tokenize(line)
        toks = [t.lower() for t in toks if t.lower() not in stop_words]
        tokens.extend(toks)

    return tokens


pos_lines = list(df[df.label == 1].headline)

pos_tokens = process_text(pos_lines)
pos_freq = nltk.FreqDist(pos_tokens)

pos_freq.most_common(20)

y_val = [x[1] for x in pos_freq.most_common()]

fig = plt.figure(figsize=(10, 5))
plt.plot(y_val)

plt.xlabel("Words")
plt.ylabel("Frequency")
plt.title("Word Frequency Distribution (Positive)")
plt.show()

y_final = []
for i, k, z, t in zip(y_val[0::4], y_val[1::4], y_val[2::4], y_val[3::4]):
    y_final.append(math.log(i + k + z + t))

x_val = [math.log(i + 1) for i in range(len(y_final))]

fig = plt.figure(figsize=(10, 5))

plt.xlabel("Words (Log)")
plt.ylabel("Frequency (Log)")
plt.title("Word Frequency Distribution (Positive)")
plt.plot(x_val, y_final)
plt.show()

neg_lines = list(df2[df2.label == -1].headline)

neg_tokens = process_text(neg_lines)
neg_freq = nltk.FreqDist(neg_tokens)

neg_freq.most_common(20)

y_val = [x[1] for x in neg_freq.most_common()]

fig = plt.figure(figsize=(10, 5))
plt.plot(y_val)

plt.xlabel("Words")
plt.ylabel("Frequency")
plt.title("Word Frequency Distribution (Negative)")
plt.show()

y_final = []

for i, k, z in zip(y_val[0::3], y_val[1::3], y_val[2::3]):
    if i + k + z == 0:
        break
    y_final.append(math.log(i + k + z))

x_val = [math.log(i + 1) for i in range(len(y_final))]

fig = plt.figure(figsize=(10, 5))

plt.xlabel("Words (Log)")
plt.ylabel("Frequency (Log)")
plt.title("Word Frequency Distribution (Negative)")
plt.plot(x_val, y_final)
plt.show()
