from classifier import Classifier
from codecs import open
import time
from flask import Flask, render_template, request

app = Flask(__name__)
classifier = Classifier()

@app.route("/predict", methods = ["POST", "GET"])
def index_page(text = "", prediction_message = ""):
    if request.method == "POST":
        text = request.form["text"]
        prediction_message = classifier.get_result_message(text)

    return render_template('simple_page.html', text = text, prediction_message = prediction_message)


if __name__ == "__main__":
    app.run(host = '0.0.0.0', port = 80, debug = True)
