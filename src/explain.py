from transformers import pipeline


summarizer = pipeline("text2text-generation", model="google/flan-t5-base")

def explain_sarcasm(text):
    prompt = f"Explain why this sentence is sarcastic: {text}"
    output = summarizer(prompt, max_length=60, do_sample=False)
    return output[0]['generated_text']