import re
from joblib import load
import pandas as pd
from matplotlib import pyplot as plt
from nltk import word_tokenize, WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.metrics import roc_auc_score, roc_curve, auc
import numpy as np

# Load the dataset
df = pd.read_csv('C:\\Users\\USER\\Desktop\\PRA\\RedditCarBrand_SentimentAnalysis\\scrape_data\\mazda.csv')

# Load models
svm_classifier = load('C:\\Users\\USER\\Desktop\\PRA\\RedditCarBrand_SentimentAnalysis\\model_build\\mazda'
                      '\\mazda_svm_classifier.joblib')
gbm_classifier = load('C:\\Users\\USER\\Desktop\\PRA\\RedditCarBrand_SentimentAnalysis\\model_build\\mazda'
                      '\\mazda_gbm_classifier.joblib')
mlp_classifier = load('C:\\Users\\USER\\Desktop\\PRA\\RedditCarBrand_SentimentAnalysis\\model_build\\mazda'
                      '\\mazda_mlp_classifier.joblib')

# Load the TF-IDF vectorizer used during training
vectorizer = load('C:\\Users\\USER\\Desktop\\PRA\\RedditCarBrand_SentimentAnalysis\\model_build\\mazda'
                  '\\mazda_tfidf_vectorizer.joblib')

# Vectorize the text data using the loaded TF-IDF vectorizer
X_vectorized = vectorizer.transform(df['headline'])

# Manual mapping for contractions
contraction_mapping = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he had",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it had",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she had",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so is",
    "that'd": "that had",
    "that'd've": "that would have",
    "that'll": "that will",
    "that'll've": "that will have",
    "that's": "that is",
    "there'd": "there had",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they had",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we had",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'alls": "you alls",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you had",
    "you'd've": "you would have",
    "you'll": "you you will",
    "you'll've": "you you will have",
    "you're": "you are",
    "you've": "you have",
    "I'm": "I am"
}


# Function to replace contractions
def replace_contractions(text):
    for contraction, expansion in contraction_mapping.items():
        text = text.replace(contraction, expansion)
    return text


# Manual mapping for emojis
emoji_mapping = {
    "😊": "happy",
    "😄": "happy",
    "😃": "happy",
    "😁": "happy",
    "😆": "happy",
    "😍": "joyful",
    "😂": "laughing",
    "😅": "laughing",
    "😉": "playful",
    "😋": "playful",
    "😎": "cool",
    "😏": "smirking",
    "😒": "displeased",
    "😌": "relieved",
    "😳": "embarrassed",
    "😴": "sleepy",
    "😞": "disappointed",
    "😖": "frustrated",
    "😟": "worried",
    "😕": "confused",
    "😮": "surprised",
    "😧": "anguished",
    "😦": "frowning",
    "😨": "fearful",
    "😰": "anxious",
    "😢": "tearful",
    "😭": "weeping",
    "😱": "terrified",
    "😩": "exhausted",
    "😵": "dizzy",
    "😷": "sick",
    "🤒": "sick",
    "🤕": "injured",
    "🤢": "nauseated",
    "🤮": "vomiting",
    "🤧": "sneezing",
    "😈": "mischievous",
    "👿": "angry",
    "💀": "dead",
    "☠️": "dead",
    "😻": "adoring",
    "😽": "kissing",
    "👻": "playful",
    "👽": "alienated",
    "👾": "alien",
    "🤖": "robotic",
    "💩": "silly",
    "👏": "applauding",
    "🙌": "celebrating",
    "🙏": "praying",
    "👍": "thumbs up",
    "👎": "thumbs down",
    "👌": "ok hand",
    "✌️": "peace",
    "🤞": "crossed fingers",
    "🤘": "rock on",
    "🤙": "call me",
    "🖐️": "raised hand",
    "✋": "raised hand",
    "👋": "waving",
    "🤚": "raised back of hand",
    "👊": "fist bump",
    "✊": "raised fist",
    "🤛": "left fist bump",
    "🤜": "right fist bump",
    "💪": "muscle",
    "🙋": "raising hand",
    "🙇": "bowing",
    "🤸": "cartwheeling",
    "🤼": "wrestling",
    "🤺": "fencing",
    "🤾": "handballing",
    "🤹": "juggling",
    "🧘": "yoga",
    ":grinning_face_with_big_eyes:": "happy",
    ":sad_face:": "sad",
    ":smiling_face_with_open_mouth:": "joy",
    ":rolling_on_the_floor_laughing:": "laughter",
    ":grinning_squinting_face:": "happiness",
    ":grinning_face_with_sweat:": "excitement",
    ":grinning_face:": "smile",
    ":beaming_face_with_smiling_eyes:": "pleasure",
    ":grinning_face_with_star_eyes:": "gratitude",
    ":grinning_face_with_tear:": "tears of joy",
    ":winking_face:": "wink",
    ":slightly_smiling_face:": "slight smile",
    ":upside_down_face:": "flipped",
    ":relieved_face:": "relieved",
    ":smiling_face_with_heart-eyes:": "love",
    ":smiling_face_with_smiling_eyes:": "affection",
    ":star-struck:": "star-struck",
    ":kissing_face:": "kiss",
    ":kissing_face_with_closed_eyes:": "close eye kiss",
    ":kissing_face_with_smiling_eyes:": "kiss with smile",
    ":face_throwing_a_kiss:": "face throwing a kiss",
    ":face_blowing_a_kiss:": "face blowing a kiss",
    ":face_savoring_food:": "savor",
    ":face_with_hand_over_mouth:": "surprise",
    ":face_with_monocle:": "monocle",
    ":nerd_face:": "nerd",
    ":thinking_face:": "thinking",
    ":zipper-mouth_face:": "zipper mouth",
    ":face_with_raised_eyebrow:": "raised eyebrow",
    ":face_with_one_eyebrow_raised:": "one eyebrow raised",
    ":confounded_face:": "confusion",
    ":persevere:": "persevere",
    ":confused:": "confused",
    ":worried:": "worried",
    ":slightly_frowning_face:": "slight frown",
    ":frowning_face:": "frown",
    ":face_with_steam_from_nose:": "steam from nose",
    ":pouting_face:": "pout",
    ":anguished:": "anguish",
    ":grimacing_face:": "grimace",
    ":face_with_open_mouth:": "open mouth",
    ":hushed:": "hush",
    ":no_mouth:": "no mouth",
    ":neutral_face:": "neutral",
    ":expressionless:": "expressionless",
    ":unamused:": "unamused",
    ":roll_eyes:": "roll eyes",
    ":thinking:": "thinking",
    ":lying_face:": "lie",
    ":grim_face:": "grim",
    ":zany_face:": "zany",
    ":face_vomiting:": "vomit",
    ":exploding_head:": "explode",
    ":cowboy_hat_face:": "cowboy hat",
    ":partying_face:": "party",
    ":disguised_face:": "disguise",
    ":ghost:": "ghost",
    ":alien:": "alien",
    ":robot_face:": "robot",
    ":smiley_cat:": "smiley cat",
    ":smile_cat:": "smile cat",
    ":joy_cat:": "joy cat",
    ":heart_eyes_cat:": "heart eyes cat",
    ":smirk_cat:": "smirk cat",
    ":kissing_cat:": "kissing cat",
    ":screaming_cat:": "screaming cat",
    ":crying_cat_face:": "crying cat",
    ":pouting_cat:": "pouting cat",
    ":see_no_evil:": "see no evil",
    ":hear_no_evil:": "hear no evil",
    ":speak_no_evil:": "speak no evil",
    ":kiss:": "kiss",
    ":love_letter:": "love letter",
    ":cupid:": "cupid",
    ":gift_heart:": "gift heart",
    ":sparkling_heart:": "sparkling heart",
    ":heartpulse:": "heartpulse",
    ":heartbeat:": "heartbeat",
    ":broken_heart:": "broken heart",
    ":two_hearts:": "two hearts",
    ":revolving_hearts:": "revolving hearts",
    ":heart_decoration:": "heart decoration",
    ":purple_heart:": "purple heart",
    ":yellow_heart:": "yellow heart",
    ":green_heart:": "green heart",
    ":blue_heart:": "blue heart",
    ":orange_heart:": "orange heart",
    ":black_heart:": "black heart"
}


# Function to replace emojis
def replace_emojis(text):
    for emoji_code, replacement in emoji_mapping.items():
        text = text.replace(emoji_code, replacement)
    return text


def preprocess_text(text):
    text = replace_contractions(text.lower())

    # Remove numbers and special characters
    text = re.sub('[^a-zA-Z]', ' ', text)

    # Convert emojis and emoticons to words
    text = replace_emojis(text)

    # Tokenize, remove stopwords, and lemmatize
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


def predict_proba_from_csv():
    try:
        # Load data
        data = pd.read_csv('C:\\Users\\USER\\Desktop\\PRA\\RedditCarBrand_SentimentAnalysis\\scrape_data\\mazda.csv')

        # Check if 'headline' column exists
        if 'headline' not in data.columns:
            raise ValueError("CSV file does not contain 'headline' column.")

        # Preprocess the 'Headline' column
        data['headline'] = data['headline'].apply(preprocess_text)

        # Get predicted probabilities for each class
        svm_probabilities = svm_classifier.predict_proba(X_vectorized)
        gbm_probabilities = gbm_classifier.predict_proba(X_vectorized)
        mlp_probabilities = mlp_classifier.predict_proba(X_vectorized)

        # Average the probabilities from different models
        avg_probabilities = (svm_probabilities + gbm_probabilities + mlp_probabilities) / 3

        return avg_probabilities, data['sentiment']

    except FileNotFoundError:
        print(f"Error: File not found.")
    except Exception as e:
        print(f"Error: {str(e)}")


def plot_roc_auc(y_true, y_pred_proba, classes):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # Compute ROC curve and ROC area for each class
    for i in range(len(classes)):
        fpr[i], tpr[i], _ = roc_curve((y_true == classes[i]).astype(int), y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    for i in range(len(classes)):
        plt.plot(fpr[i], tpr[i], label=f'Class {classes[i]} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-AUC Curve for Mazda')
    plt.legend(loc="lower right")
    plt.show()

    # Print ROC-AUC scores
    for i, sentiment_class in enumerate(classes):
        print(f"ROC-AUC score for class {sentiment_class}: {roc_auc[i]}")


# Predictions for the classes
predictions_proba, y_true = predict_proba_from_csv()  # Assuming you have a function for this

# Extracting the sentiment classes
sentiment_classes = sorted(df['sentiment'].unique())

# Plot ROC-AUC curve
plot_roc_auc(y_true, predictions_proba, sentiment_classes)
