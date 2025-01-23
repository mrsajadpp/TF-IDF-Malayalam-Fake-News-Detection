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
news_example = "കേരളത്തിൽ പമ്പ താലൂക്കിൽ ഭൂകമ്പം ഉണ്ടായതായി റിപ്പോർട്ട്"
fake_news = "വിളക്കുതീ ഉണ്ടാക്കുന്നതിന് പുതിയ നിയമങ്ങൾ നടപ്പാക്കുന്നു"
print(f"Prediction: {predict(fake_news)}")
