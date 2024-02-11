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
def land():
    return render_template('land.html')


@app.route('/plot')
def plot():
    return render_template('plot.html')


@app.route('/table')
def table():
    return render_template('table.html')


@app.route('/menu')
def menu():
    return render_template('menu.html')


@app.route('/brand')
def brand():
    return render_template('brand.html')


@app.route('/spec')
def spec():
    return render_template('spec.html')


def get_csv_filenames(subreddit):
    prefix = f'{subreddit.lower()}'
    return f'{prefix}.csv'


@app.route('/brand_detail', methods=['POST'])
def brand_detail():
    subreddit_name = request.form['subreddit_name']
    csv_directory = Path('scrape_data')
    # Load the saved csv based on subreddit
    model_filename = get_csv_filenames(subreddit_name)
    print(model_filename)

    csv_path = csv_directory / model_filename
    data = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=(5, 5))

    counts = data.label.value_counts(normalize=True) * 100

    sns.barplot(x=counts.index, y=counts, ax=ax)

    ax.set_xticklabels(['Negative', 'Neutral', 'Positive'])
    ax.set_ylabel("Percentage")

    # Save the plot as a PNG image
    plot_buffer_sentiment = BytesIO()
    plt.savefig(plot_buffer_sentiment, format='png')
    plot_buffer_sentiment.seek(0)
    sentiment_plot_data_uri = base64.b64encode(plot_buffer_sentiment.read()).decode('utf-8')

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

    pos_lines = list(data[data.label == 1].headline)
    neg_lines = list(data[data.label == 0].headline)  # Add this line to get negative lines

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

    return render_template('display_brand.html',
                           sentiment_plot=sentiment_plot_data_uri,
                           positive_plot=positive_plot_data_uri,
                           negative_plot=negative_plot_data_uri)


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

    # Generate plots and related output
    fig, ax = plt.subplots()
    # Example plot
    df['sentiment'].value_counts().plot(kind='bar', ax=ax)
    # Save plot to a BytesIO object
    plot_buffer = BytesIO()
    plt.savefig(plot_buffer, format='png')
    plot_buffer.seek(0)
    plot_data_uri = base64.b64encode(plot_buffer.read()).decode('utf-8')

    fig, ax = plt.subplots()
    # Example plot
    df['sentiment'].value_counts().plot(kind='pie', ax=ax)
    # Save plot to a BytesIO object
    plot_buffer = BytesIO()
    plt.savefig(plot_buffer, format='png')
    plot_buffer.seek(0)
    plot_data_uri_pie = base64.b64encode(plot_buffer.read()).decode('utf-8')

    # Convert DataFrame to HTML table
    table_html = df.to_html()
    subreddit_name_title_case = subreddit_name.title()
    keyword_title_case = keyword.title()
    return render_template('display_spec.html', plot_data_uri=plot_data_uri, plot_data_uri_pie=plot_data_uri_pie,
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
