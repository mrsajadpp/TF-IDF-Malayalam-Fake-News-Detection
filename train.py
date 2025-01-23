import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv("malayalam_news.csv")

# Encode labels
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])  # real: 1, fake: 0

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# Text vectorization using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000, analyzer='word', stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train the model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Evaluate the model
y_pred = model.predict(X_test_tfidf)
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Save the model and vectorizer
joblib.dump(model, "model/fake_news_model.pkl")
joblib.dump(vectorizer, "model/tfidf_vectorizer.pkl")
joblib.dump(label_encoder, "model/label_encoder.pkl")

# def predict_fake_news(news_text):
#     # Preprocess and vectorize the input text
#     news_tfidf = vectorizer.transform([news_text])
#     prediction = model.predict(news_tfidf)
#     return label_encoder.inverse_transform(prediction)[0]

# # Test with example
# news_example = "ഇതൊരു വ്യാജവാർത്തയാണ്"
# print(f"Prediction: {predict_fake_news(news_example)}")
