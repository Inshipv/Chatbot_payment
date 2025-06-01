from flask import Flask, render_template, request, jsonify
import torch
import random
import json
from model import NeuralNet
from utils import bag_of_words, tokenize
import nltk

# Download NLTK 'punkt' tokenizer resource quietly
nltk.download('punkt', quiet=True)

app = Flask(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model and intent data
with open("intents.json", 'r') as f:
    intents = json.load(f)

FILE = "chatbot_model.pth"
data = torch.load(FILE)

input_size = data["input_size"]
output_size = data["output_size"]
hidden_size = data["hidden_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

# Initialize model
model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    message = request.form["message"]
    
    # Tokenize and get the model's response
    sentence_tokens = tokenize(message)
    X = bag_of_words(sentence_tokens, all_words)
    X = torch.from_numpy(X).float().unsqueeze(0).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    response = ""
    if prob.item() > 0.7:
        for intent in intents["intents"]:
            if intent["tag"] == tag:
                response = random.choice(intent['responses'])
    else:
        response = "Sorry, I don't understand that. Could you rephrase?"

    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
