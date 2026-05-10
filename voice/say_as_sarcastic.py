from flask import Flask, render_template, request, redirect
from src.bert_predict import predict_bert
import csv

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        sentence = request.form["sentence"]
        prediction = predict_bert(sentence)
        result = {
            "text": prediction["text"],
            "sarcasm_en": "Sarcastic" if prediction["is_sarcastic"] else "Not Sarcastic",
            "confidence": prediction["confidence"]
        }
    return render_template("index.html", result=result)

@app.route("/feedback", methods=["POST"])
def feedback():
    sentence = request.form["sentence"]
    prediction = request.form["prediction"]
    confidence = request.form["confidence"]
    user_feedback = request.form["feedback"]

    with open("feedback_log.csv", mode="a", newline='', encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([sentence, prediction, confidence, user_feedback])

    return redirect("/")

@app.route("/quiz", methods=["GET", "POST"])
def quiz():
    # (Mevcut quiz mantığın buraya gelir)
    return render_template("quiz.html", result={"text": "Quiz complete! Refresh to restart."})

if __name__ == "__main__":
    app.run(debug=True)
