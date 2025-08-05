import joblib

target_names = ['alt.atheism',
 'comp.graphics',
 'comp.os.ms-windows.misc',
 'comp.sys.ibm.pc.hardware',
 'comp.sys.mac.hardware',
 'comp.windows.x',
 'misc.forsale',
 'rec.autos',
 'rec.motorcycles',
 'rec.sport.baseball',
 'rec.sport.hockey',
 'sci.crypt',
 'sci.electronics',
 'sci.med',
 'sci.space',
 'soc.religion.christian',
 'talk.politics.guns',
 'talk.politics.mideast',
 'talk.politics.misc',
 'talk.religion.misc']

class Classifier(object):
    def __init__(self):
        self.vectorizer = joblib.load("news_vectorizer_dump_tfidf.bin")
        self.model = joblib.load("news_model_dump_rf.bin")
        self.target_names = target_names
    
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
