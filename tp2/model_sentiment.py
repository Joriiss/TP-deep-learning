import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


def load_txt_data(file_path, label):
    with open(file_path, 'r', encoding='utf-8') as f:
        
        content = f.read()
        reviews = [r.strip() for r in content.split('\n') if len(r.strip()) > 10]
    return pd.DataFrame({'review': reviews, 'sentiment': label})


df_pos = load_txt_data("TrainingDataPositive.txt", "positive")
df_neg = load_txt_data("TrainingDataNegative.txt", "negative")
df_train = pd.concat([df_pos, df_neg], ignore_index=True)


model_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 2))),
    ('nb', MultinomialNB())
])


model_pipeline.fit(df_train['review'], df_train['sentiment'])


joblib.dump(model_pipeline, 'modele_custom_nb.joblib')