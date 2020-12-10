from flask import *
import numpy as np
import pandas as pd
import joblib

classifier = joblib.load('spam_ham_message.pkl')
vectorizer = joblib.load('transform.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods = ['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = vectorizer.transform(data).toarray()
        prediction = classifier.predict(vect)

    return render_template('result.html', prediction_message = prediction)


if __name__ == '__main__':
    app.run(debug= True)
