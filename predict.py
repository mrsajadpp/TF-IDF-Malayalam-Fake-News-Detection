from fastapi import FastAPI, Query
import joblib

app = FastAPI()

# Load the model and vectorizer
model = joblib.load("model/fake_news_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")
label_encoder = joblib.load("model/label_encoder.pkl")

@app.post("/predict/")
def predict(news_text: str = Query(..., description="Enter Malayalam news text")):
    news_tfidf = vectorizer.transform([news_text])
    prediction = model.predict(news_tfidf)
    result = label_encoder.inverse_transform(prediction)[0]
    return {"text": news_text, "prediction": result}

# Test with example
news_example = "വരുന്ന ആഴ്ച ലോകം അവസാനിക്കും"
print(f"Prediction: {predict(news_example)}")
