import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('C:\\Users\\USER\\Desktop\\PRA\\RedditCarBrand_SentimentAnalysis\\scrape_data\\mazda.csv')

# Display the head of the DataFrame (first few rows)
print("Head of the DataFrame:")
print(df.head())

# Display descriptive statistics of the DataFrame
print("\nDescriptive Statistics:")
print(df.describe())
