import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import joblib

class FakeNewsDetector:

    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=1, min_df=1)
        self.pac = PassiveAggressiveClassifier()

    def preprocessing(self):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.df['text'], self.df['label'], test_size=0.1, random_state=7)
        self.tfidf_train = self.tfidf_vectorizer.fit_transform(self.x_train)

    def train_model(self):
        self.pac.fit(self.tfidf_train, self.y_train)

    def save_objects(self):
        joblib.dump(self.pac, 'pac_model.joblib')
        joblib.dump(self.tfidf_vectorizer, 'tfidf_vectorizer.joblib')

        self.test_df = pd.DataFrame(data={'text': self.x_test, 'label': self.y_test})
        self.test_df.to_csv('unknown.csv')
    

def train_model_and_save(csv_file):
    fake_news_detector = FakeNewsDetector(csv_file)
    fake_news_detector.preprocessing()
    fake_news_detector.train_model()
    fake_news_detector.save_objects()

if __name__ == '__main__':
    csv_file = 'news.csv'
    train_model_and_save(csv_file)
