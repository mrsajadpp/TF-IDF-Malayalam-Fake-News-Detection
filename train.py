import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_preprocess_data(filepath):
    """
    Load CSV and perform initial preprocessing
    """
    df = pd.read_csv(filepath)
    
    # Encode labels
    label_encoder = LabelEncoder()
    df['label_encoded'] = label_encoder.fit_transform(df['label'])
    
    return df, label_encoder

def train_and_evaluate(X, y, class_weights):
    """
    Train and evaluate the Logistic Regression model with class weights
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Vectorize text
    vectorizer = TfidfVectorizer(max_features=5000, analyzer='word', stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Train the model with class weights
    model = LogisticRegression(max_iter=1000, random_state=42, class_weight=class_weights)
    model.fit(X_train_tfidf, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test_tfidf)
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    return model, vectorizer

def main():
    # Load data
    filepath = "malayalam_news.csv"
    df, label_encoder = load_and_preprocess_data(filepath)
    
    # Calculate class weights
    class_counts = df['label_encoded'].value_counts()
    class_weights = {0: class_counts[1] / class_counts[0], 1: class_counts[0] / class_counts[1]}
    
    # Train and evaluate the model with class weights
    model, vectorizer = train_and_evaluate(df['text'], df['label_encoded'], class_weights)
    
    # Save model and components
    joblib.dump(model, "model/weighted_fake_news_model.pkl")
    joblib.dump(vectorizer, "model/weighted_tfidf_vectorizer.pkl")
    joblib.dump(label_encoder, "model/weighted_label_encoder.pkl")

if __name__ == "__main__":
    main()