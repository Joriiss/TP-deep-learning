import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib

nltk.download('vader_lexicon')

df_reviews = pd.read_csv("./TestReviews.csv")
analyzer = SentimentIntensityAnalyzer()

def classify_mood(text):
    
    
    scores = analyzer.polarity_scores(text)
    compound = scores["compound"]

    
    if compound >= 0.05:
        return "positive"
    elif compound <= -0.05:
        return "negative"
    else:
        return "neutre"


df_reviews["sentiment"] = df_reviews["review"].apply(classify_mood)


from transformers import pipeline

sentiment_pipeline = pipeline("sentiment-analysis")

def safe_transformer(text):
    if not isinstance(text, str) or len(text.strip()) == 0:
        return "neutral"
    
    return text[:512]


reviews_list = df_reviews["review"].apply(safe_transformer).tolist()
results = sentiment_pipeline(reviews_list)

df_reviews["sentiment_transformer"] = [res['label'].lower() for res in results]

nb_model = joblib.load('modele_custom_nb.joblib')
df_reviews["sentiment_nb_custom"] = nb_model.predict(df_reviews["review"].astype(str))


df_reviews.to_csv("sentiment.csv", index=False, encoding="utf-8")