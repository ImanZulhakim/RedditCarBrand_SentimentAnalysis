from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from joblib import load
from collections import Counter
import praw
import time
import seaborn as sns
from pprint import pprint
from pathlib import Path
import nltk
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)


@app.route('/')
def menu():
    return render_template('menu.html')


@app.route('/brand')
def brand():
    return render_template('brand.html')


@app.route('/spec')
def spec():
    return render_template('spec.html')


@app.route('/compare')
def compare():
    return render_template('compare.html')


@app.route('/compare_results')
def compare_results():
    render_template('compare_results.html')


def get_csv_filenames(subreddit):
    prefix = f'{subreddit.lower()}'
    return f'{prefix}.csv'


@app.route('/brand_compare', methods=['POST'])
def brand_compare():
    subreddit_name1 = request.form['subreddit_name1']
    subreddit_name2 = request.form['subreddit_name2']

    csv_directory = Path('scrape_data')

    # Load the saved csv based on subreddit 1
    model_filename1 = get_csv_filenames(subreddit_name1)
    csv_path1 = csv_directory / model_filename1
    data1 = pd.read_csv(csv_path1)

    # Load the saved csv based on subreddit 2
    model_filename2 = get_csv_filenames(subreddit_name2)
    csv_path2 = csv_directory / model_filename2
    data2 = pd.read_csv(csv_path2)

    # Plot 1 - Sentiment Analysis for subreddit 1
    fig, ax = plt.subplots(figsize=(5, 5))
    counts1 = data1.label.value_counts(normalize=True) * 100

    # Filter the data for negative, neutral and positive sentiment.
    negative_data = data1[data1['label'] == -1]
    neutral_data = data1[data1['label'] == 0]
    positive_data = data1[data1['label'] == 1]

    # Compute the percentage of negative, neutral and positive sentiment
    percentage_negative = (len(negative_data) / len(data1)) * 100
    percentage_neutral = (len(neutral_data) / len(data1)) * 100
    percentage_positive = (len(positive_data) / len(data1)) * 100

    palette = {'Negative': '#dc3545', 'Neutral': '#adb5bd', 'Positive': '#198754'}
    sns.barplot(x=counts1.index, y=counts1, ax=ax, palette=palette.values())
    ax.set_xticks(range(3))  # Set the positions of ticks
    ax.set_xticklabels(['Negative', 'Neutral', 'Positive'])
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Percentage")

    # Annotating the plot with the percentage of negative, neutral, and positive sentiment
    ax.text(0, counts1.max() * 0.03, f'{percentage_negative:.2f}%', color='black', ha='center', fontweight='bold')
    ax.text(1, counts1.max() * 0.03, f'{percentage_neutral:.2f}%', color='black', ha='center', fontweight='bold')
    ax.text(2, counts1.max() * 0.03, f'{percentage_positive:.2f}%', color='black', ha='center', fontweight='bold')

    plot_buffer_sentiment1 = BytesIO()
    plt.savefig(plot_buffer_sentiment1, format='png')
    plot_buffer_sentiment1.seek(0)
    sentiment_plot_data_uri1 = base64.b64encode(plot_buffer_sentiment1.read()).decode('utf-8')
    plt.close()

    # Plot 2 - Sentiment Analysis for subreddit 2
    fig, ax = plt.subplots(figsize=(5, 5))
    counts2 = data2.label.value_counts(normalize=True) * 100

    # Filter the data for negative, neutral and positive sentiment.
    negative_data = data2[data2['label'] == -1]
    neutral_data = data2[data2['label'] == 0]
    positive_data = data2[data2['label'] == 1]

    # Compute the percentage of negative, neutral and positive sentiment
    percentage_negative = (len(negative_data) / len(data2)) * 100
    percentage_neutral = (len(neutral_data) / len(data2)) * 100
    percentage_positive = (len(positive_data) / len(data2)) * 100

    sns.barplot(x=counts2.index, y=counts2, ax=ax, palette=palette.values())
    ax.set_xticks(range(3))  # Set the positions of ticks
    ax.set_xticklabels(['Negative', 'Neutral', 'Positive'])
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Percentage")

    # Annotating the plot with the percentage of negative, neutral, and positive sentiment
    ax.text(0, counts2.max() * 0.03, f'{percentage_negative:.2f}%', color='black', ha='center', fontweight='bold')
    ax.text(1, counts2.max() * 0.03, f'{percentage_neutral:.2f}%', color='black', ha='center', fontweight='bold')
    ax.text(2, counts2.max() * 0.03, f'{percentage_positive:.2f}%', color='black', ha='center', fontweight='bold')

    plot_buffer_sentiment2 = BytesIO()
    plt.savefig(plot_buffer_sentiment2, format='png')
    plot_buffer_sentiment2.seek(0)
    sentiment_plot_data_uri2 = base64.b64encode(plot_buffer_sentiment2.read()).decode('utf-8')
    plt.close()

    # Pie Chart 1 - Sentiment Analysis for subreddit 1
    fig, ax = plt.subplots()
    data1['label'] = data1['label'].map({-1: 'Negative', 0: 'Neutral', 1: 'Positive'})
    ax.set_xlabel("Sentiment")
    data1['label'].value_counts().plot(kind='pie', ax=ax, colors=[palette[label] for label in data1['label'].unique()])
    labels = [f'{label} ({sizes:.2f}%)' for label, sizes in
              zip(data1['label'].unique(), (data1['label'].value_counts(normalize=True) * 100))]
    ax.legend(labels, loc="best")
    plot_buffer_pie1 = BytesIO()
    plt.savefig(plot_buffer_pie1, format='png')
    plot_buffer_pie1.seek(0)
    plot_data_uri_pie1 = base64.b64encode(plot_buffer_pie1.read()).decode('utf-8')
    plt.close()

    # Pie Chart 2 - Sentiment Analysis for subreddit 2
    fig, ax = plt.subplots()
    data2['label'] = data2['label'].map({-1: 'Negative', 0: 'Neutral', 1: 'Positive'})
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Percentage")
    data2['label'].value_counts().plot(kind='pie', ax=ax, colors=[palette[label] for label in data2['label'].unique()])
    labels = [f'{label} ({sizes:.2f}%)' for label, sizes in
              zip(data2['label'].unique(), (data2['label'].value_counts(normalize=True) * 100))]
    ax.legend(labels, loc="best")
    plot_buffer_pie2 = BytesIO()
    plt.savefig(plot_buffer_pie2, format='png')
    plot_buffer_pie2.seek(0)
    plot_data_uri_pie2 = base64.b64encode(plot_buffer_pie2.read()).decode('utf-8')
    plt.close()

    subreddit_title1 = subreddit_name1.upper()
    subreddit_title2 = subreddit_name2.upper()
    # Initialize dictionaries to store sentiment values for each subreddit
    sentiment_values1 = {'Positive': 0, 'Neutral': 0, 'Negative': 0}
    sentiment_values2 = {'Positive': 0, 'Neutral': 0, 'Negative': 0}

    # Calculate sentiment values for subreddit 1
    sentiment_counts1 = data1.label.value_counts(normalize=True) * 100
    for sentiment, value in sentiment_counts1.items():
        sentiment_values1[sentiment] = value

    # Calculate sentiment values for subreddit 2
    sentiment_counts2 = data2.label.value_counts(normalize=True) * 100
    for sentiment, value in sentiment_counts2.items():
        sentiment_values2[sentiment] = value

    data_table1 = pd.read_csv(csv_path1)
    print(data_table1.head())
    print("Positive headlines:\n")
    pprint(list(data_table1[data_table1['label'] == 1].headline)[:5], width=200)

    print("\nNeutral headlines:\n")
    pprint(list(data_table1[data_table1['label'] == 0].headline)[:5], width=200)

    print("\nNegative headlines:\n")
    pprint(list(data_table1[data_table1['label'] == -1].headline)[:5], width=200)

    brand_positive_headlines1 = list(data_table1[data_table1['label'] == 1].headline)[:10]
    brand_neutral_headlines1 = list(data_table1[data_table1['label'] == 0].headline)[:10]
    brand_negative_headlines1 = list(data_table1[data_table1['label'] == -1].headline)[:10]

    data_table2 = pd.read_csv(csv_path2)
    print(data_table2.head())
    print("Positive headlines:\n")
    pprint(list(data_table2[data_table2['label'] == 1].headline)[:5], width=200)

    print("\nNeutral headlines:\n")
    pprint(list(data_table2[data_table2['label'] == 0].headline)[:5], width=200)

    print("\nNegative headlines:\n")
    pprint(list(data_table2[data_table2['label'] == -1].headline)[:5], width=200)

    brand_positive_headlines2 = list(data_table2[data_table2['label'] == 1].headline)[:10]
    brand_neutral_headlines2 = list(data_table2[data_table2['label'] == 0].headline)[:10]
    brand_negative_headlines2 = list(data_table2[data_table2['label'] == -1].headline)[:10]

    # Pass the dictionaries to the template
    return render_template('compare_results.html',
                           subreddit_name1=subreddit_name1,
                           subreddit_name2=subreddit_name2,
                           subreddit_title1=subreddit_title1,
                           subreddit_title2=subreddit_title2,
                           sentiment_plot1=sentiment_plot_data_uri1,
                           sentiment_plot_pie1=plot_data_uri_pie1,
                           sentiment_plot2=sentiment_plot_data_uri2,
                           sentiment_plot_pie2=plot_data_uri_pie2,
                           sentiment_values1=sentiment_values1,
                           sentiment_values2=sentiment_values2,
                           brand_positive_headlines1=brand_positive_headlines1,
                           brand_neutral_headlines1=brand_neutral_headlines1,
                           brand_negative_headlines1=brand_negative_headlines1,
                           brand_positive_headlines2=brand_positive_headlines2,
                           brand_neutral_headlines2=brand_neutral_headlines2,
                           brand_negative_headlines2=brand_negative_headlines2
                           )


@app.route('/brand_detail', methods=['POST'])
def brand_detail():
    subreddit_name = request.form['subreddit_name']
    csv_directory = Path('scrape_data')
    # Load the saved csv based on subreddit
    model_filename = get_csv_filenames(subreddit_name)
    print(model_filename)

    csv_path = csv_directory / model_filename
    data = pd.read_csv(csv_path)

    # Prepare the nltk tokenizer and stopwords
    tokenizer = RegexpTokenizer(r'\w+')
    stop_words = stopwords.words('english')

    fig, ax = plt.subplots(figsize=(5, 5))

    counts = data.label.value_counts(normalize=True) * 100

    palette = {'Negative': '#dc3545', 'Neutral': '#adb5bd', 'Positive': '#198754'}

    sns.barplot(x=counts.index, y=counts, ax=ax, palette=palette.values())

    ax.set_xticks(range(3))  # Set the positions of ticks

    ax.set_xticklabels(['Negative', 'Neutral', 'Positive'])
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Percentage")

    # Save the plot as a PNG image
    plot_buffer_sentiment = BytesIO()
    plt.savefig(plot_buffer_sentiment, format='png')
    plot_buffer_sentiment.seek(0)
    sentiment_plot_data_uri = base64.b64encode(plot_buffer_sentiment.read()).decode('utf-8')
    plt.close()

    fig, ax = plt.subplots()
    # Mapping of labels
    sentiment_map = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}
    # Replace numerical labels with corresponding sentiment labels
    data['label'] = data['label'].map(sentiment_map)

    # Create the pie chart
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Percentage")
    data['label'].value_counts().plot(kind='pie', ax=ax, colors=[palette[label] for label in data['label'].unique()])
    # Save plot to a BytesIO object
    plot_buffer = BytesIO()
    plt.savefig(plot_buffer, format='png')
    plot_buffer.seek(0)
    plot_data_uri_pie = base64.b64encode(plot_buffer.read()).decode('utf-8')
    plt.close()

    data_frequency = pd.read_csv(csv_path)
    # Prepare the nltk tokenizer and stopwords
    tokenizer = RegexpTokenizer(r'\w+')
    stop_words = stopwords.words('english')

    def process_text_brand(headlines):
        tokens = []
        for line in headlines:
            toks = tokenizer.tokenize(line)
            toks = [t.lower() for t in toks if t.lower() not in stop_words]
            tokens.extend(toks)

        return tokens

    pos_lines = list(data_frequency[data_frequency.label == 1].headline)
    neg_lines = list(data_frequency[data_frequency.label == 0].headline)  # Add this line to get negative lines

    pos_tokens = process_text_brand(pos_lines)
    pos_freq = nltk.FreqDist(pos_tokens)

    pos_freq.most_common(20)

    y_val = [x[1] for x in pos_freq.most_common()]

    fig = plt.figure(figsize=(5, 5))
    plt.plot(y_val)

    plt.xlabel("Words")
    plt.ylabel("Frequency")
    plt.title("Word Frequency Distribution (Positive)")

    # Save the positive sentiment plot as a PNG image
    plot_buffer_positive = BytesIO()
    plt.savefig(plot_buffer_positive, format='png')
    plot_buffer_positive.seek(0)
    positive_plot_data_uri = base64.b64encode(plot_buffer_positive.read()).decode('utf-8')
    plt.close()

    neg_tokens = process_text_brand(neg_lines)  # Process negative lines
    neg_freq = nltk.FreqDist(neg_tokens)  # Frequency distribution of negative tokens

    # Plot negative sentiment
    fig = plt.figure(figsize=(5, 5))
    plt.plot([x[1] for x in neg_freq.most_common(20)])

    plt.xlabel("Words")
    plt.ylabel("Frequency")
    plt.title("Word Frequency Distribution (Negative)")

    # Save the negative sentiment plot as a PNG image
    plot_buffer_negative = BytesIO()
    plt.savefig(plot_buffer_negative, format='png')
    plot_buffer_negative.seek(0)
    negative_plot_data_uri = base64.b64encode(plot_buffer_negative.read()).decode('utf-8')
    plt.close()

    data_table = pd.read_csv(csv_path)
    print(data_table.head())
    print("Positive headlines:\n")
    pprint(list(data_table[data_table['label'] == 1].headline)[:5], width=200)

    print("\nNeutral headlines:\n")
    pprint(list(data_table[data_table['label'] == 0].headline)[:5], width=200)

    print("\nNegative headlines:\n")
    pprint(list(data_table[data_table['label'] == -1].headline)[:5], width=200)

    brand_positive_headlines = list(data_table[data_table['label'] == 1].headline)[:10]
    brand_neutral_headlines = list(data_table[data_table['label'] == 0].headline)[:10]
    brand_negative_headlines = list(data_table[data_table['label'] == -1].headline)[:10]
    brand_positive_url = list(data_table[data_table['label'] == 1].url)[:10]
    brand_neutral_url = list(data_table[data_table['label'] == 0].url)[:10]
    brand_negative_url = list(data_table[data_table['label'] == -1].url)[:10]

    subreddit_title_case = subreddit_name.title()

    # Zip the headlines and URLs
    brand_positive_headlines_urls = zip(brand_positive_headlines, brand_positive_url)
    brand_neutral_headlines_urls = zip(brand_neutral_headlines, brand_neutral_url)
    brand_negative_headlines_urls = zip(brand_negative_headlines, brand_negative_url)

    return render_template('display_brand.html',
                           sentiment_plot=sentiment_plot_data_uri,
                           sentiment_plot_pie=plot_data_uri_pie,
                           positive_plot=positive_plot_data_uri,
                           negative_plot=negative_plot_data_uri,
                           subreddit_title_case=subreddit_title_case,
                           brand_positive_headlines_urls=brand_positive_headlines_urls,
                           brand_neutral_headlines_urls=brand_neutral_headlines_urls,
                           brand_negative_headlines_urls=brand_negative_headlines_urls
                           )


@app.route('/analyze', methods=['POST'])
def analyze():
    subreddit_name = request.form['subreddit_name']
    keyword = request.form['keyword']

    # Your Reddit scraping code here
    filtered_headlines = scrape_reddit(subreddit_name, keyword)

    # Convert the list of headlines to a DataFrame
    df = pd.DataFrame({'Headline': filtered_headlines})

    # Define the CSV file name including the keyword
    csv_file_name = f'{subreddit_name}_{keyword}.csv'

    # Save the DataFrame to a CSV file with the specified name
    df.to_csv(csv_file_name, index=False)

    df.head()

    # Perform sentiment prediction using majority vote
    predictions = predict_from_csv(csv_file_name, subreddit_name)
    df['sentiment'] = predictions
    df.to_csv('predicted.csv', index=False)

    print(df.head())

    data = pd.read_csv('predicted.csv')
    print(data.head())
    print("Positive headlines:\n")
    pprint(list(data[data['sentiment'] == 1].Headline)[:5], width=200)

    print("\nNeutral headlines:\n")
    pprint(list(data[data['sentiment'] == 0].Headline)[:5], width=200)

    print("\nNegative headlines:\n")
    pprint(list(data[data['sentiment'] == -1].Headline)[:5], width=200)

    positive_headlines = list(data[data['sentiment'] == 1].Headline)[:10]
    neutral_headlines = list(data[data['sentiment'] == 0].Headline)[:10]
    negative_headlines = list(data[data['sentiment'] == -1].Headline)[:10]

    # Filter the data for negative, neutral and positive sentiment.
    negative_data = data[data['sentiment'] == -1]
    neutral_data = data[data['sentiment'] == 0]
    positive_data = data[data['sentiment'] == 1]

    # Compute the percentage of negative, neutral and positive sentiment
    percentage_negative = (len(negative_data) / len(data)) * 100
    percentage_neutral = (len(neutral_data) / len(data)) * 100
    percentage_positive = (len(positive_data) / len(data)) * 100

    fig, ax = plt.subplots(figsize=(5, 5))
    counts = data.sentiment.value_counts(normalize=True) * 100
    palette = {'Negative': '#dc3545', 'Neutral': '#adb5bd', 'Positive': '#198754'}

    sns.barplot(x=counts.index, y=counts, ax=ax, palette=palette.values())
    ax.set_xticks(range(3))  # Set the positions of ticks

    # Annotating the plot with the percentage of negative, neutral, and positive sentiment
    ax.text(0, counts.max() * 0.03, f'{percentage_negative:.2f}%', color='black', ha='center', fontweight='bold')
    ax.text(1, counts.max() * 0.03, f'{percentage_neutral:.2f}%', color='black', ha='center', fontweight='bold')
    ax.text(2, counts.max() * 0.03, f'{percentage_positive:.2f}%', color='black', ha='center', fontweight='bold')

    ax.set_xticklabels(['Negative', 'Neutral', 'Positive'])
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Percentage")

    # Save the plot as a PNG image
    plot_buffer_sentiment = BytesIO()
    plt.savefig(plot_buffer_sentiment, format='png')
    plot_buffer_sentiment.seek(0)
    sentiment_plot_data_uri = base64.b64encode(plot_buffer_sentiment.read()).decode('utf-8')
    plt.close()

    # Plot a pie graph
    fig, ax = plt.subplots()
    sentiment_map = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}
    # Replace numerical labels with corresponding sentiment labels
    data['sentiment'] = data['sentiment'].map(sentiment_map)

    # Create the pie chart
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Percentage")
    data['sentiment'].value_counts().plot(kind='pie', ax=ax,
                                          colors=[palette[sentiment] for sentiment in data['sentiment'].unique()])
    labels = [f'{sentiment} ({sizes:.2f}%)' for sentiment, sizes in
              zip(data['sentiment'].unique(), (data['sentiment'].value_counts(normalize=True) * 100))]
    ax.legend(labels, loc="best", bbox_to_anchor=(0.85, 0.9))

    # Save plot to a BytesIO object
    plot_buffer = BytesIO()
    plt.savefig(plot_buffer, format='png')
    plot_buffer.seek(0)
    plot_data_uri_pie = base64.b64encode(plot_buffer.read()).decode('utf-8')
    plt.close()

    # Convert DataFrame to HTML table
    table_html = df.to_html()
    subreddit_name_title_case = subreddit_name.title()
    keyword_title_case = keyword.title()
    return render_template('display_spec.html', sentiment_plot_data_uri=sentiment_plot_data_uri,
                           plot_data_uri_pie=plot_data_uri_pie,
                           positive_headlines=positive_headlines,
                           neutral_headlines=neutral_headlines, negative_headlines=negative_headlines,
                           table_html=table_html, subr=subreddit_name_title_case, keyword=keyword_title_case)


def scrape_reddit(subreddit_name, keyword):
    # Initialize Reddit API with your credentials
    reddit = praw.Reddit(client_id='UwQGu_BZV3plC9jLCXaRTg',
                         client_secret='pIR4KDYFE0cxjBh73acT7voeSEKm7g',
                         user_agent='LapSent')

    # Create a list to store the filtered headlines
    filtered_headlines = []

    # Set the time range for submissions (from today to one year ago)
    end_time = int(time.time())  # Current epoch time
    start_time = end_time - 63072000  # 2 year ago

    # Iterate through the submissions in the subreddit within the time range
    for submission in reddit.subreddit(subreddit_name).search(f'{keyword}', time_filter='year', limit=None):
        # Check if the keyword is in the title
        if keyword.lower() in submission.title.lower():
            # Add the headline to the list
            filtered_headlines.append(submission.title)

    return filtered_headlines


# Function to construct filenames for trained models based on subreddit
def get_model_filenames(subreddit):
    prefix = f'{subreddit.lower()}_'
    return {
        'tfidf_vectorizer': f'{prefix}tfidf_vectorizer.joblib',
        'svm_classifier': f'{prefix}svm_classifier.joblib',
        'gbm_classifier': f'{prefix}gbm_classifier.joblib',
        'mlp_classifier': f'{prefix}mlp_classifier.joblib'
    }


def predict_from_csv(csv_file_path, subreddit):
    try:
        models_directory = Path(f'model_build/{subreddit}')

        # Load the saved models based on subreddit
        model_filenames = get_model_filenames(subreddit)
        svm_classifier = load(models_directory / model_filenames['svm_classifier'])
        gbm_classifier = load(models_directory / model_filenames['gbm_classifier'])
        mlp_classifier = load(models_directory / model_filenames['mlp_classifier'])

        print(models_directory / model_filenames['svm_classifier'])
        # Load the TF-IDF vectorizer used during training
        vectorizer = load(models_directory / model_filenames['tfidf_vectorizer'])

        # Read the CSV file
        df = pd.read_csv(csv_file_path)

        # Check if 'Headline' column exists
        if 'Headline' not in df.columns:
            raise ValueError("CSV file does not contain 'Headline' column.")

        # Preprocess the 'Headline' column
        df['Headline'] = df['Headline'].apply(preprocess_text)

        # Vectorize the text data using the loaded TF-IDF vectorizer
        X_vectorized = vectorizer.transform(df['Headline'])

        # Make predictions using the loaded models
        svm_predictions = svm_classifier.predict(X_vectorized)
        gbm_predictions = gbm_classifier.predict(X_vectorized)
        mlp_predictions = mlp_classifier.predict(X_vectorized)

        # Perform majority voting
        majority_votes = []
        for i in range(len(df)):
            votes = [svm_predictions[i], gbm_predictions[i], mlp_predictions[i]]
            majority_vote = Counter(votes).most_common(1)[0][0]
            majority_votes.append(majority_vote)

        return majority_votes

    except FileNotFoundError:
        print(f"Error: File not found at path {csv_file_path}.")
    except Exception as e:
        print(f"Error: {str(e)}")


def preprocess_text(text):
    if isinstance(text, str):
        tokens = word_tokenize(text.lower())
        filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]
        lemmatizer = WordNetLemmatizer()
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    else:
        tokens = word_tokenize(str(text))
        filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]
        lemmatizer = WordNetLemmatizer()
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    processed_text = ' '.join(lemmatized_tokens)

    return processed_text


if __name__ == '__main__':
    app.run(debug=True)
