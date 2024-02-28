import numpy as np
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Activation
from tensorflow.keras.optimizers import RMSprop
from flask import Flask, request, jsonify

app = Flask(__name__)

# Read the data
text_ds = pd.read_csv("microscope.csv")
text = list(text_ds.text.values)
joined_text = " ".join(text)

# Tokenization
tokenizer = RegexpTokenizer(r"\w+")
tokens = tokenizer.tokenize(joined_text.lower())

# Unique tokens
unique_tokens = np.unique(tokens)
unique_token_index = {token: idex for idex, token in enumerate(unique_tokens)}

# Model parameters
n_words = 10

input_words = []
next_words = []

# Prepare input and output data
for i in range(len(tokens) - n_words):
    input_words.append(tokens[i:i + n_words])
    next_words.append(tokens[i + n_words])

x = np.zeros((len(input_words), n_words, len(unique_tokens)), dtype=bool)
y = np.zeros((len(input_words), len(unique_tokens)), dtype=bool)

for i, words in enumerate(input_words):
    for j, word in enumerate(words):
        x[i, j, unique_token_index[word]] = 1
    y[i, unique_token_index[next_words[i]]] = 1

# Loading the model
model = load_model('my_model_prediction_text_generate.keras')

# Function to predict the next word
def generate_text(input_text, text_length, creativity=3):
    word_sequence = input_text.split()
    current = 0
    for _ in range(text_length):
        sub_sequence = " ".join(word_sequence[current:current + n_words])
        try:
            top_predictions = predict_next_word(sub_sequence, creativity)
            choice = select_best_prediction(top_predictions, word_sequence[current:])
        except:
            choice = np.random.choice(unique_tokens)
        # Filter out repeated words
        if choice != word_sequence[-1] and choice not in word_sequence[current:]:
            word_sequence.append(choice)
        current += 1
    
    generated_text = " ".join(word_sequence)
    if generated_text[-1] not in ['.', '!', '?']:
        generated_text += '.'
    
    return generated_text

def select_best_prediction(predictions, context):
    # Choose the prediction that fits the context best
    for pred in predictions:
        if pred in context:
            return pred
    # If none of the predictions fit the context, choose randomly
    return np.random.choice(list(predictions.keys()))

def predict_next_word(input_text, creativity):
    input_text = input_text.lower()
    x = np.zeros((1, n_words, len(unique_tokens)))

    for i, word in enumerate(input_text.split()):
        if word in unique_token_index:
            x[0, i, unique_token_index[word]] = 1

    predictions = model.predict(x)[0]
    top_n_indices = np.argpartition(predictions, -creativity)[-creativity:]
    top_predictions = {unique_tokens[idx]: predictions[idx] for idx in top_n_indices}
    
    return top_predictions

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    input_text = data['input_text']
    text_length = data['text_length']
    creativity = data.get('creativity', 3)
    generated_text = generate_text(input_text, text_length, creativity)
    return jsonify({'generated_text': generated_text})

if __name__ == '__main__':
    app.run(debug=True)
