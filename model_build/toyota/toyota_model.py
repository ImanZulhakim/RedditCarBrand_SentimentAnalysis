from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
from joblib import dump
import re

# Load the dataset
df = pd.read_csv('C:\\Users\\USER\\Desktop\\PRA\\RedditCarBrand_SentimentAnalysis\\scrape_data\\toyota.csv')


# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df['headline'], df['sentiment'], test_size=0.2, random_state=42)

# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Check if the test set is empty
if X_test_vectorized.shape[0] == 0:
    print("Error: Test set is empty. Check your train-test split or vectorization process.")
    exit()

# Support Vector Machine (SVM) Classifier
svm_params = {'C': [0.1, 1, 10], 'gamma': [0.1, 1, 10], 'kernel': ['rbf', 'linear']}
svm_classifier = GridSearchCV(SVC(probability=True), svm_params, cv=5)
svm_classifier.fit(X_train_vectorized, y_train)
y_pred_svm = svm_classifier.predict(X_test_vectorized)

# Gradient Boosting Machines (GBM) Classifier
gbm_params = {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.5]}
gbm_classifier = GridSearchCV(GradientBoostingClassifier(), gbm_params, cv=5)
gbm_classifier.fit(X_train_vectorized, y_train)
y_pred_gbm = gbm_classifier.predict(X_test_vectorized)

# Neural Network (backpropagation) Classifier
mlp_params = {'hidden_layer_sizes': [(150, 100, 50), (120, 80, 40), (100, 50, 30)], 'max_iter': [500, 1000],
              'activation': ['tanh', 'relu'],
              'solver': ['adam'],
              'alpha': [0.0001, 0.05],
              'learning_rate': ['constant', 'adaptive']}
mlp_classifier = GridSearchCV(MLPClassifier(), mlp_params, cv=5)
mlp_classifier.fit(X_train_vectorized, y_train)
y_pred_mlp = mlp_classifier.predict(X_test_vectorized)

# Print the best parameters for SVM
print("Best Parameters for SVM:", svm_classifier.best_params_)

# Print the best parameters for GBM
print("Best Parameters for GBM:", gbm_classifier.best_params_)

# Print the best parameters for MLP
print("Best Parameters for MLP:", mlp_classifier.best_params_)

# Evaluate accuracy on the test set for each classifier
accuracy_svm = accuracy_score(y_test, y_pred_svm)
accuracy_gbm = accuracy_score(y_test, y_pred_gbm)
accuracy_mlp = accuracy_score(y_test, y_pred_mlp)

print("\nAccuracy on Test Set (SVM):", accuracy_svm)
print("\nAccuracy on Test Set (GBM):", accuracy_gbm)
print("\nAccuracy on Test Set (MLP):", accuracy_mlp)

# Save trained models
dump(vectorizer, 'toyota_tfidf_vectorizer.joblib')
dump(svm_classifier, 'toyota_svm_classifier.joblib')
dump(gbm_classifier, 'toyota_gbm_classifier.joblib')
dump(mlp_classifier, 'toyota_mlp_classifier.joblib')
