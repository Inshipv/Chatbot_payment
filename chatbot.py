# chatbot.py
import random
import torch
from model import NeuralNet
from utils import bag_of_words, tokenize
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()

bot_name = "PayBot"
print(f"{bot_name} is ready! Type 'quit' to exit.")

while True:
    sentence = input("You: ")
    if sentence.lower() == "quit":
        break

    sentence_tokens = tokenize(sentence)
    X = bag_of_words(sentence_tokens, all_words)
    X = torch.from_numpy(X).float().unsqueeze(0).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.7:
        for intent in intents["intents"]:
            if intent["tag"] == tag:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        print(f"{bot_name}: I'm not sure I understand. Can you rephrase?")
