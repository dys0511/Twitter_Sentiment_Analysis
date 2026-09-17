import os
import re
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

def clean_tweet(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    return text.strip()

def preprocess_and_save():
    os.makedirs("data", exist_ok=True)
    
    csv_path = "data/training.1600000.processed.noemoticon.csv"
    if not os.path.exists(csv_path):
        print(f"Error: Please download Sentiment140 and place the CSV file at {csv_path}")
        return

    columns = ["target", "ids", "date", "flag", "user", "text"]
    print("Loading dataset...")
    df = pd.read_csv(csv_path, encoding="latin-1", names=columns)
    
    # Remap Sentiment140 target (0 = Negative, 4 = Positive -> 1)
    df["target"] = df["target"].replace(4, 1)
    
    # Sample 50,000 tweets for cross-model efficiency
    df_sampled = df.groupby("target", group_keys=False).apply(lambda x: x.sample(25000, random_state=42))
    
    print("Cleaning tweets...")
    df_sampled["clean_text"] = df_sampled["text"].apply(clean_tweet)
    
    X = df_sampled["clean_text"]
    y = df_sampled["target"].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Extracting TF-IDF features...")
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words="english")
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print("Saving processed data artifacts to data/...")
    save_npz("data/X_train.npz", X_train_tfidf)
    save_npz("data/X_test.npz", X_test_tfidf)
    np.save("data/y_train.npy", y_train)
    np.save("data/y_test.npy", y_test)
    joblib.dump(vectorizer, "data/tfidf_vectorizer.pkl")
    
    print("Preprocessing completed successfully.")

if __name__ == "__main__":
    preprocess_and_save()