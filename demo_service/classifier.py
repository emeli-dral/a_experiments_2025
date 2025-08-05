import joblib
from sklearn.datasets import fetch_20newsgroups


class Classifier(object):
    def __init__(self):
        self.vectorizer = joblib.load("news_vectorizer_dump_tfidf.pkl")
        self.model = joblib.load("news_model_dump_rf.pkl")
        self.target_names = fetch_20newsgroups(subset = 'test').target_names
    
    def get_name_by_label(self, label: str) -> str:
        try:
            return self.target_names[label]
        except:
            return "label error"

    def predict_text(self, text: str) -> str:
        try:
            vectorized = self.vectorizer.transform([text])
            return self.model.predict(vectorized)[0] 
        except:
            print("prediction error")
            return None 

    def get_result_message(self, text: str) -> str:
        prediction = self.predict_text(text)
        return self.get_name_by_label(prediction)
