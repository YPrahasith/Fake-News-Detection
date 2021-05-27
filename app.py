import joblib

from flask import Flask, request, render_template, redirect, url_for
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST', ])
def result():
    if request.method == 'POST':
        news_text = request.form['news_text']
        pac = joblib.load('pac_model.joblib')
        tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')

        tfidf_test = tfidf_vectorizer.transform((news_text, ))
        result = pac.predict(tfidf_test)[0]

        textcolor = 'red' if result == 'FAKE' else 'green'

        return render_template('result.html', news_text=news_text, textcolor=textcolor, result=result)
