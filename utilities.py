import re
import os
import json

import nltk
import numpy as np
from gensim.models import Word2Vec
from nltk import word_tokenize
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout, GlobalMaxPooling1D, Conv2D, \
    MaxPooling2D, Flatten
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
import keras
from tensorflow.keras.callbacks import EarlyStopping
from wordcloud import WordCloud

nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

essential_stopwords = {'not', 'no', 'very', 'only', 'but'}
stop_words = set(stopwords.words('english')) - essential_stopwords
lemmatizer = WordNetLemmatizer()


def showWordCloud(list_of_string):
    combined_text = " ".join([" ".join(row) for row in list_of_string])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(combined_text)

    # Display the generated word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')  # Turn off axis
    plt.show()


def tokenize_and_remove_stopwords(sentences):
    return [
        [word for word in word_tokenize(sentence.lower()) if word not in stop_words and word.isalpha()]
        for sentence in sentences
    ]


def get_average_vector_for_sentences(sentences, model):
    word_vectors = []
    for sentence in sentences:
        sentence_vectors = [model.wv[word] for word in sentence if word in model.wv]
        word_vectors.extend(sentence_vectors)
    if len(word_vectors) == 0:  # Return zero vector if no words found in the model
        return np.zeros(model.vector_size)
    return np.mean(word_vectors, axis=0)


def textProcessing(tweet_list):
    # Convert the list to a string
    tweet = ' '.join(tweet_list)
    # tweet = re.sub(r'[^\w\s]', '', tweet)  #remove punctuations and characters

    # Tokenization
    tokens = nltk.word_tokenize(tweet.lower())  # Convert text to tokens

    # Remove single-character tokens (except meaningful ones like 'i' and 'a')
    tokens = [word for word in tokens if len(word) > 1]

    # Remove stopwords
    tweet = [word for word in tokens if word not in stop_words]

    # Lemmatization
    tweet = [lemmatizer.lemmatize(word, pos='v') for word in tweet]

    # Join words back into a single string
    tweet = ' '.join(tweet)
    return tweet


def read_json_from_folder(folder_path):
    json_data_list = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.json'):  # Only process JSON files
            file_path = os.path.join(folder_path, file_name)

            # Open and read each JSON file
            with open(file_path, 'r', encoding='utf-8') as json_file:
                try:
                    print(f"Loading file {file_path}")
                    data = json.load(json_file)  # Load the JSON data

                    # Append the JSON data to the list
                    json_data_list.append(data)

                except json.JSONDecodeError:
                    print(f"Error reading {file_name}")

    return json_data_list


def printClassificationReport(model, X_train, X_test, y_train, y_test):
    predictions = model.predict(X_train)
    predictions = (predictions > 0.5).astype(int)
    print("Train Data Classification report")
    print(classification_report(y_train, predictions))

    # Test Data Results
    predictions = model.predict(X_test)
    predictions = (predictions > 0.5).astype(int)
    print("Test Data Classification report")
    print(classification_report(y_test, predictions))


def runSVCModel(X_train, X_test, y_train, y_test):
    model = LinearSVC()
    model.fit(X_train, y_train)
    printClassificationReport(model, X_train, X_test, y_train, y_test)
    return model


def runRandomForestClassifier(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    printClassificationReport(model, X_train, X_test, y_train, y_test)
    return model


def runDecisionTreeClassifier(X_train, X_test, y_train, y_test):
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    printClassificationReport(model, X_train, X_test, y_train, y_test)
    return model


def runMultiLayerPerceptronWithBCE(X_train, X_test, y_train, y_test):
    model = keras.Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dropout(0.3),
        Dense(8, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')  # Output layer for binary classification
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test),
                        callbacks=[early_stopping])
    loss, accuracy = model.evaluate(X_test, y_test)
    train_loss, train_accuracy = model.evaluate(X_train, y_train)
    print(f"Training Loss: {train_loss}, Training Accuracy: {train_accuracy}")
    print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")
    printClassificationReport(model, X_train, X_test, y_train, y_test)
    return history


def runMultiLayerPerceptronWithCCE(X_train, X_test, y_train, y_test):
    model = keras.Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dropout(0.3),
        Dense(8, activation='relu'),
        Dropout(0.2),
        Dense(4, activation='softmax')  # Output layer for binary classification
    ])
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test),
                        callbacks=[early_stopping])
    loss, accuracy = model.evaluate(X_test, y_test)
    train_loss, train_accuracy = model.evaluate(X_train, y_train)
    print(f"Training Loss: {train_loss}, Training Accuracy: {train_accuracy}")
    print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

    y_pred_train = model.predict(X_train)
    y_pred_train_classes = np.argmax(y_pred_train, axis=1)
    print("Classification report for train set")
    print(classification_report(y_train, y_pred_train_classes, target_names=[str(i) for i in range(1, 5)]))

    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    print("Classification report for test set")
    print(classification_report(y_test, y_pred_classes, target_names=[str(i) for i in range(1, 5)]))

    return history

def runMultiLayerPerceptronWithSCCEUsingW2V(X_train, X_test, y_train, y_test, vocab_length, max_len, embedding_dim, embedding_matrix):
    model = keras.Sequential([
        Embedding(input_dim=vocab_length,
                  output_dim=embedding_dim,
                  weights=[embedding_matrix],
                  input_length=max_len,
                  trainable=False),
        Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l2(0.0002))),
        Dropout(0.3),
        Bidirectional(LSTM(32, return_sequences=True, kernel_regularizer=l2(0.0002))),
        Dropout(0.3),
        GlobalMaxPooling1D(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(4, activation='softmax')  # Output layer for binary classification
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test),
                        callbacks=[early_stopping])
    loss, accuracy = model.evaluate(X_test, y_test)
    train_loss, train_accuracy = model.evaluate(X_train, y_train)
    print(f"Training Loss: {train_loss}, Training Accuracy: {train_accuracy}")
    print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

    y_pred_train = model.predict(X_train)
    y_pred_train_classes = np.argmax(y_pred_train, axis=1)
    y_true_train = np.argmax(y_test, axis=1)
    print("Classification report for train set")
    print(classification_report(y_true_train, y_pred_train_classes, target_names=[str(i) for i in range(1, 5)]))


    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)
    print("Classification report for test set")
    print(classification_report(y_true, y_pred_classes, target_names=[str(i) for i in range(1, 5)]))

    return history


def runSVCMultiLabel(X_train, X_test, y_train, y_test):
    svc = SVC(kernel='linear', probability=True)
    # Use One-vs-Rest strategy
    model = OneVsRestClassifier(svc)
    # Train the model
    model.fit(X_train, y_train)
    # Make predictions
    y_pred_test = model.predict(X_test)
    # Make predictions on the training set
    y_pred_train = model.predict(X_train)
    # Calculate accuracy
    accuracy_train = accuracy_score(y_train, y_pred_train)
    accuracy_test = accuracy_score(y_test, y_pred_test)
    f1_train = f1_score(y_train, y_pred_train, average='macro')
    f1_test = f1_score(y_test, y_pred_test, average='macro')

    # Print accuracy and F1 scores
    print(f"Training Accuracy: {accuracy_train:.2f}")
    print(f"Test Accuracy: {accuracy_test:.2f}")
    print(f"Training F1 Score (macro): {f1_train:.2f}")
    print(f"Test F1 Score (macro): {f1_test:.2f}")


def runMultiLabelClassification(X_train, X_test, y_train, y_test):
    # Step 5: Build the neural network model
    model = Sequential([
        Dense(128, input_shape=(X_train.shape[1],), activation='relu'),
        Dropout(0.3),  # Prevent overfitting
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(y_train.shape[1], activation='softmax')  # 22 output nodes with sigmoid for multi-label
    ])

    # Compile the model with binary cross-entropy loss for multi-label classification
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=["accuracy"])
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test),
                        callbacks=[early_stopping])
    loss, accuracy = model.evaluate(X_test, y_test)
    train_loss, train_accuracy = model.evaluate(X_train, y_train)
    print(f"Training Loss: {train_loss}, Training Accuracy: {train_accuracy}")
    print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")
    return history


def runMultiLabelClassificationWithBatchNormalization(X_train, X_test, y_train, y_test):
    # Step 5: Build the neural network model
    model = Sequential([
        Dense(256, input_shape=(X_train.shape[1],), activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(y_train.shape[1], activation='relu')  # Output layer remains the same
    ])

    # Compile the model with binary cross-entropy loss for multi-label classification
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=["accuracy"])
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test),
                        callbacks=[early_stopping])
    loss, accuracy = model.evaluate(X_test, y_test)
    train_loss, train_accuracy = model.evaluate(X_train, y_train)
    print(f"Training Loss: {train_loss}, Training Accuracy: {train_accuracy}")
    print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")
    return history


def runMultiClassification(X_train, X_test, y_train, y_test):
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=l2(0.0002)))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(y_train.shape[1], activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=["accuracy"])
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test),
                        callbacks=[early_stopping])
    loss, accuracy = model.evaluate(X_test, y_test)
    train_loss, train_accuracy = model.evaluate(X_train, y_train)
    print(f"Training Loss: {train_loss}, Training Accuracy: {train_accuracy}")
    print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")
    return history


def runLSTM(X_train, X_test, y_train, y_test):
    model = Sequential()
    model.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(y_train.shape[1], activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=["accuracy"])
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test),
                        callbacks=[early_stopping])
    loss, accuracy = model.evaluate(X_test, y_test)
    train_loss, train_accuracy = model.evaluate(X_train, y_train)
    print(f"Training Loss: {train_loss}, Training Accuracy: {train_accuracy}")
    print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")
    return history


def runLSTMBinaryClassification(X_train, X_test, y_train, y_test):
    model = Sequential()
    model.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(y_train.shape[1], activation='sigmoid'))

    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=["accuracy"])
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test),
                        callbacks=[early_stopping])
    loss, accuracy = model.evaluate(X_test, y_test)
    train_loss, train_accuracy = model.evaluate(X_train, y_train)
    print(f"Training Loss: {train_loss}, Training Accuracy: {train_accuracy}")
    print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")
    printClassificationReport(model, X_train, X_test, y_train, y_test)
    return history


def plotTrainAccuracyAndLoss(modelhistory):
    # Visualize the training and validation metrics
    plt.figure(figsize=(12, 5))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(modelhistory.history['accuracy'], label='Train Accuracy')
    plt.plot(modelhistory.history['val_accuracy'], label='Test Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(modelhistory.history['loss'], label='Train Loss')
    plt.plot(modelhistory.history['val_loss'], label='Test Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()


def convertStringListToString(text_list):
    return ' '.join(text_list)


def load_glove_embeddings(glove_file_path, tokenizer, embedding_dim=100):
    embeddings_index = {}
    with open(glove_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    # Create the embedding matrix
    embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, embedding_dim))
    for word, idx in tokenizer.word_index.items():
        if word in embeddings_index:
            embedding_matrix[idx] = embeddings_index[word]

    return embedding_matrix


def load_word2vec_embeddings(sentences, tokenizer, embedding_dim=300):
    # Train a Word2Vec model
    word2vec_model = Word2Vec(sentences, vector_size=embedding_dim, window=5, min_count=1, workers=4)

    # Initialize embedding matrix with zeros
    vocab_size = len(tokenizer.word_index) + 1
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    # Retrieve Word2Vec embeddings for each word
    for word, idx in tokenizer.word_index.items():
        if word in word2vec_model.wv.key_to_index:
            embedding_matrix[idx] = word2vec_model.wv[word]

    return embedding_matrix