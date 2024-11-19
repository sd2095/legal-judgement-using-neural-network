import pandas as pd
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from utilities import read_json_from_folder, runSVCModel, runRandomForestClassifier, \
    runDecisionTreeClassifier, textProcessing, tokenize_and_remove_stopwords, \
    get_average_vector_for_sentences, runMultiLayerPerceptronWithBCE, plotTrainAccuracyAndLoss, \
    runLSTMBinaryClassification
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import numpy as np

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

# Adding new column
df['VIOLATION_STATUS'] = df['VIOLATED_ARTICLES'].apply(lambda x: 1 if len(x) > 0 else 0)

# Processing text columns
print("Processing text column to remove the stop words and lemmatization")
df['TEXT'] = df['TEXT'].apply(textProcessing)


print("Using TFIDF Vectorizer to transform TEXT column values")
vect = TfidfVectorizer()
X = vect.fit_transform(df.TEXT)


## Implementing Word 2 Vec
# print("Started tokenization")
# df['TOKENIZED_TEXT'] = df['TEXT'].apply(tokenize_and_remove_stopwords)
# all_sentences = [sentence for sentence_list in df['TOKENIZED_TEXT'] for sentence in sentence_list]
# print("Started word 2 vec model")
# word2vec_model = Word2Vec(sentences=all_sentences, vector_size=500, window=12, min_count=1, sg=1)
# df['TOKENIZED_VECTOR'] = df['TOKENIZED_TEXT'].apply(lambda x: get_average_vector_for_sentences(x, word2vec_model))
#
# X = np.array(df['TOKENIZED_VECTOR'].tolist())

y = df['VIOLATION_STATUS'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

## Violation Status binary class classification

# print("Running SVC Model")
# svcModel = runSVCModel(X_train, X_test, y_train, y_test)
# print("Saving SVC Model")
# torch.save(svcModel, 'binary_classfication_svc_with_tfidf.pt')
#
# print("Running SVC Model using Word2Vec")
# svcModel = runSVCModel(X_train, X_test, y_train, y_test)
# print("Saving SVC Model using Word2Vec")
# torch.save(svcModel, 'binary_classfication_svc_with_w2v.pt')
#
# print("Running Random Forest Classifier Model")
# randomForestModel = runRandomForestClassifier(X_train, X_test, y_train, y_test)
# print("Saving Random Forest Classifier Model")
# torch.save(randomForestModel, 'binary_classfication_random_forest_with_tfidf.pt')
#
# print("Running Random Forest Classifier Model Using Word2Vec")
# randomForestModel = runRandomForestClassifier(X_train, X_test, y_train, y_test)
# print("Saving Random Forest Classifier Model Using Word2Vec")
# torch.save(randomForestModel, 'binary_classfication_random_forest_with_w2v.pt')
#
# print("Running Decision Tree Classifier Model")
# model = runDecisionTreeClassifier(X_train, X_test, y_train, y_test)
# print("Saving Decision Tree Classifier Model")
# torch.save(model, 'binary_classfication_decision_tree_with_tfidf.pt')
#
# print("Running Decision Tree Classifier Model Using Word2Vec")
# model = runDecisionTreeClassifier(X_train, X_test, y_train, y_test)
# print("Saving Decision Tree Classifier Model Using Word2Vec")
# torch.save(model, 'binary_classfication_decision_tree_with_w2v.pt')

# print("Running Multi Layer Perceptron using Word 2 Vec")
# modelHistory = runMultiLayerPerceptronWithBCE(X_train, X_test, y_train, y_test)
# plotTrainAccuracyAndLoss(modelHistory)
# print("Saving MLP Model using Word2Vec")
# torch.save(modelHistory, 'binary_classfication_mlp_with_w2v.pt')

# print("Running Multi Layer Perceptron using TFIDF")
# modelHistory = runMultiLayerPerceptronWithBCE(X_train, X_test, y_train, y_test)
# plotTrainAccuracyAndLoss(modelHistory)
# print("Saving MLP Model using TFIDF")
# torch.save(modelHistory, 'binary_classfication_mlp_with_tfidf.pt')
