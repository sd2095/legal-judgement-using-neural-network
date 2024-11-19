import numpy as np
from gensim.models import Word2Vec
from keras.src.utils import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from utilities import read_json_from_folder, textProcessing
import pandas as pd
import torch
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from utilities import stop_words
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout, GlobalMaxPooling1D, Conv2D, \
    MaxPooling2D, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import BatchNormalization



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

# Processing text columns
print("Processing text column to remove the stop words and lemmatization")
df['TEXT'] = df['TEXT'].apply(textProcessing)

# neural_X = df.TEXT
# neural_y = df.IMPORTANCE.astype(int).sub(1)
#
# X_neural_train, X_neural_test, y_neural_train, y_neural_test = train_test_split(neural_X, neural_y, test_size=0.2,
#                                                                                 stratify=neural_y, random_state=42)
#
# texts = X_neural_train.tolist()
# texts = [text.split() for text in texts]
# w2v = Word2Vec(texts, vector_size=200, window=5, workers=7, epochs=100, min_count=5)
#
# max_len=5000
#
# tokenizer = Tokenizer()
# tokenizer.fit_on_texts(X_neural_train)
# vocab_length = len(tokenizer.word_index) + 1
#
#
# X_neural_train = tokenizer.texts_to_sequences(X_neural_train)
# X_neural_test = tokenizer.texts_to_sequences(X_neural_test)
#
# X_neural_train = pad_sequences(X_neural_train, maxlen=max_len, padding='post')
# X_neural_test = pad_sequences(X_neural_test, maxlen=max_len, padding='post')
#
# print(vocab_length)
#
# embedding_dim = 200
# embedding_matrix = np.zeros((vocab_length, embedding_dim))
#
# for word, i in tokenizer.word_index.items():
#     if word in w2v.wv:
#         embedding_matrix[i] = w2v.wv[word]
#
# model = Sequential()
#
# # Embedding layer with the pre-trained Word2Vec weights
# model.add(Embedding(input_dim=vocab_length,
#                     output_dim=embedding_dim,
#                     weights=[embedding_matrix],
#                     input_length=max_len,  # max length of input sequences
#                     trainable=False))
# model.add(LSTM(64))
# model.add(Dropout(0.2))
# model.add(Dense(4, activation='softmax'))
# model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#
# num_epochs = 5
# history = model.fit(X_neural_train, y_neural_train, epochs=num_epochs)
#
# test_loss, test_accuracy = model.evaluate(X_neural_test, y_neural_test)
# print(f"Test Loss: {test_loss}")
# print(f"Test Accuracy: {test_accuracy}")
# y_neural_test_pred = model.predict(X_neural_test)
# print(y_neural_test_pred)


print("Using TFIDF Vectorizer to transform TEXT column values")
vect = TfidfVectorizer(min_df=0.0001, max_df=0.95, stop_words=list(stop_words), max_features=10000)
vect.fit(df.TEXT)
X = vect.transform(df.TEXT)
y = df.IMPORTANCE.astype(int).sub(1)

# ## Implementing Word 2 Vec
# print("Started word2 vec implementation")
# df['TEXT'] = df['TEXT'].apply(convertStringListToString)
#
# texts = [text.split() for text in df['TEXT']]
# w2v = Word2Vec(texts, vector_size=200, window=5, workers=7, epochs=100, min_count=5)
#
# max_len = np.max(df['TEXT'].apply(lambda x :len(x)))
# print("Max length of string ", max_len)
# tokenizer = Tokenizer()
# tokenizer.fit_on_texts(texts)
# vocab_length = len(tokenizer.word_index) + 1
#
# print("Started tokenization")
# X = tokenizer.texts_to_sequences(texts)
# X = pad_sequences(X, maxlen=max_len, padding='post')
#
# print(vocab_length)
#
# embedding_dim = 200
# embedding_matrix = np.zeros((vocab_length, embedding_dim))
#
# print("Started creating word embedding matrix")
# for word, i in tokenizer.word_index.items():
#     if word in w2v.wv:
#         embedding_matrix[i] = w2v.wv[word]


# Encode labels (importance) as categorical
# y = to_categorical(df['IMPORTANCE'].values.astype(int) - 1, num_classes=4)


# labels = np.array(df['IMPORTANCE'].values).astype(int) - 1
# y = to_categorical(labels, num_classes=4)

print("Splitting train test split")
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

model = Sequential()

# Embedding layer with the pre-trained Word2Vec weights
model = Sequential([
    Dense(256, input_shape=(X_train.shape[1],), activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(4, activation='softmax')  # Output layer remains the same
])

# Compile the model with binary cross-entropy loss for multi-label classification
model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=["accuracy"])
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test),
                    callbacks=[early_stopping])
loss, accuracy = model.evaluate(X_test, y_test)
train_loss, train_accuracy = model.evaluate(X_train, y_train)
print(f"Training Loss: {train_loss}, Training Accuracy: {train_accuracy}")
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")


predictions = model.predict(X_test)
z=np.array(np.argmax(predictions,axis=1))
z=z.reshape(len(predictions),1)
print("Test Data Classification report")
print(classification_report(y_test,z))

#
# print("Running Multi Layer Perceptron using TFIDF")
# modelHistory = runMultiLayerPerceptronWithCCE(X_train, X_test, y_train, y_test)
# plotTrainAccuracyAndLoss(modelHistory)
# print("Saving MLP Model using TFIDF")
# torch.save(modelHistory, 'multiclass_classfication_mlp_with_tfidf.pt')

# print("Running Multi Layer Perceptron using W2V")
# modelHistory = runMultiLayerPerceptronWithSCCEUsingW2V(X_train, X_test, y_train, y_test, vocab_length, max_len, embedding_dim, embedding_matrix)
# plotTrainAccuracyAndLoss(modelHistory)
# print("Saving MLP Model using W2V")
# torch.save(modelHistory, 'multiclass_classfication_mlp_with_w2v.pt')
