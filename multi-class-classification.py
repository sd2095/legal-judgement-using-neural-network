import numpy as np
from gensim.models import Word2Vec
from keras.src.utils import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.regularizers import l2

from utilities import read_json_from_folder, convertStringListToString
import pandas as pd
import torch
from tensorflow.keras.preprocessing.text import Tokenizer
from utilities import stop_words
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout, GlobalMaxPooling1D, Conv2D, \
    MaxPooling2D, Flatten
from tensorflow.keras.models import Sequential

folder_path = 'ECHR_Dataset/EN_train'
folder_path2 = 'ECHR_Dataset/EN_test'

# Read JSON files from both folders
print("Started reading json files for train data")
train_data = read_json_from_folder(folder_path)
print("Started reading json files for test data")
test_data = read_json_from_folder(folder_path2)

# Combine the data from both folders into one list
combined_data = train_data + test_data
df = pd.DataFrame(combined_data)
print("Completed framing data into df")

# Removing outliers
df['TEXT_LENGTH'] = df['TEXT'].apply(lambda x: len(' '.join(x)))
df = df[df['TEXT_LENGTH'] <= df['TEXT_LENGTH'].quantile(0.998)]

neural_X = df['TEXT'].apply(convertStringListToString)
neural_y = df.IMPORTANCE.astype(int).sub(1)

print("Train test split")
X_neural_train, X_neural_test, y_neural_train, y_neural_test = train_test_split(neural_X, neural_y, test_size=0.2,
                                                                                stratify=neural_y, random_state=42)
texts = X_neural_train.tolist()
texts = [text.split() for text in texts]
print("Implementing word 2 vec")
w2v = Word2Vec(texts, vector_size=200, window=5, workers=7, epochs=100, min_count=5)

max_len = np.max(X_neural_train.apply(lambda x: len(x)))

print("Tokenization")
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_neural_train)
vocab_length = len(tokenizer.word_index) + 1

X_neural_train = tokenizer.texts_to_sequences(X_neural_train)
X_neural_test = tokenizer.texts_to_sequences(X_neural_test)

X_neural_train = pad_sequences(X_neural_train, maxlen=max_len, padding='post')
X_neural_test = pad_sequences(X_neural_test, maxlen=max_len, padding='post')

print(vocab_length)

embedding_dim = 200
embedding_matrix = np.zeros((vocab_length, embedding_dim))

print("Embedding matrix preparation")
for word, i in tokenizer.word_index.items():
    if word in w2v.wv:
        embedding_matrix[i] = w2v.wv[word]

model = Sequential()

# Embedding layer with the pre-trained Word2Vec weights
model.add(Embedding(input_dim=vocab_length,
                    output_dim=embedding_dim,
                    weights=[embedding_matrix],
                    input_length=max_len,  # max length of input sequences
                    trainable=False))
model.add(LSTM(64))
model.add(Dropout(0.2))
model.add(Dense(4, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print("Model training")
history = model.fit(X_neural_train, y_neural_train, epochs=5)

test_loss, test_accuracy = model.evaluate(X_neural_test, y_neural_test)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")
y_neural_test_pred = model.predict(X_neural_test)
print(y_neural_test_pred)


