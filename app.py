from flask import Flask, render_template, request, redirect
from src.bert_predict import predict_bert
import pyttsx3
import threading
import csv
import os
import random

app = Flask(__name__)

quiz_sentences = [
    "I'm so excited to do this for the 100th time.",
    "Oh great, another bug in production!",
    "I love that this happened again!",
    "Wow, what a surprise!",
    "Thanks a lot for your help, really appreciated.",
    "Oh wow, this is amazing...",
    "Just perfect timing.",
    "Sure, because I have nothing better to do.",
    "This is exactly what I needed today.",
    "Fantastic! Just spilled coffee all over myself.",
    "Broken build on Friday, nice!",
    "Production bug again. Awesome!",   
    "Perfect, the internet is down again."
]
used_sentences = set()


def speak_text(text):
    try:
        local_engine = pyttsx3.init(driverName='sapi5')
        local_engine.say(text)
        local_engine.runAndWait()
        local_engine.stop()
    except Exception as e:
        print(f"TTS Error: {e}")

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        sentence = request.form.get("sentence", "").strip()
        if not sentence:
            return render_template("index.html", result=None)

        prediction = predict_bert(sentence)
        sarcasm = "EVET (alaycı)" if prediction["is_sarcastic"] else "HAYIR (değil)"
        confidence = f"{prediction['confidence']}%"
        sarcasm_en = "This sentence is sarcastic." if prediction["is_sarcastic"] else "This sentence is not sarcastic."

        result = {
            "text": sentence,
            "sarcasm": sarcasm,
            "confidence": prediction["confidence"],
            "sarcasm_en": sarcasm_en
        }

        threading.Thread(target=speak_text, args=(
            f"Sentence: {sentence}. {sarcasm_en}. Confidence: {confidence}",
        ), daemon=True).start()

    return render_template("index.html", result=result)

@app.route("/feedback", methods=["POST"])
def feedback():
    sentence = request.form.get("sentence", "")
    prediction = request.form.get("prediction", "")
    confidence = request.form.get("confidence", "")
    feedback_value = request.form.get("feedback", "")

    with open("feedback.csv", "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([sentence, prediction, confidence, feedback_value])

    return redirect("/")

@app.route("/quiz", methods=["GET", "POST"])
def quiz():
    global used_sentences
    feedback = None

    if request.method == "GET":
        if len(used_sentences) == len(quiz_sentences):
            result = {"text": "Quiz complete! Refresh to restart."}
            return render_template("quiz.html", result=result)

        remaining = list(set(quiz_sentences) - used_sentences)
        sentence = random.choice(remaining)
        used_sentences.add(sentence)

        prediction = predict_bert(sentence.strip())

        result = {
            "text": sentence,
            "correct": int(prediction["is_sarcastic"]),
            "prediction": "Sarcastic" if prediction["is_sarcastic"] else "Not Sarcastic",
            "confidence": prediction["confidence"]
        }

        threading.Thread(target=speak_text, args=(
            f"Sentence: {sentence}. Hmm... What do you think? Is this sarcastic or not?",
        ), daemon=True).start()

        return render_template("quiz.html", result=result, feedback=None)

    if request.method == "POST":
        sentence = request.form.get("current", "").strip()
        user_answer = int(request.form.get("answer", -1))
        correct_answer = int(request.form.get("correct", -1))
        confidence = request.form.get("confidence", "N/A")

        if sentence:
            prediction = predict_bert(sentence)
            is_correct = (user_answer == correct_answer)

            feedback = "Correct!" if is_correct else f"Wrong! Model predicted: {'Sarcastic' if correct_answer else 'Not Sarcastic'}"

            result = {
                "text": sentence,
                "correct": correct_answer,
                "prediction": "Sarcastic" if correct_answer else "Not Sarcastic",
                "confidence": prediction["confidence"]
            }

            threading.Thread(target=speak_text, args=(
                f"I think this is {result['prediction'].lower()}. Confidence: {result['confidence']} percent.",
            ), daemon=True).start()

            return render_template("quiz.html", result=result, feedback=feedback)

        return redirect("/quiz")

@app.route("/quiz-feedback", methods=["POST"])
def quiz_feedback():
    sentence = request.form.get("sentence", "")
    prediction = request.form.get("prediction", "")
    confidence = request.form.get("confidence", "")
    feedback_value = request.form.get("feedback", "")

    with open("feedback.csv", "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([sentence, prediction, confidence, feedback_value])

    return redirect("/quiz")

if __name__ == "__main__":
    app.run(debug=True)
