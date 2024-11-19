# mlb = MultiLabelBinarizer()
# X = vect.fit_transform(df.TEXT).toarray()
# y = df.VIOLATED_ARTICLES
# y_filled = [item if isinstance(item, list) else [] for item in y]
# y_filled = mlb.fit_transform(y_filled)
# X_train, X_test, y_train, y_test = train_test_split(X, y_filled, test_size=0.2, random_state=42)
#
# modelHistoryMLC = runMultiLabelClassification(X_train, X_test, y_train, y_test)
# plotTrainAccuracyAndLoss(modelHistoryMLC)

#modelHistoryMLB = runMultiLabelClassification(X_train_mlb, X_test_mlb, y_train_mlb, y_test_mlb)
#plotTrainAccuracyAndLoss(modelHistoryMLB)

#modelHistoryMLC = runMultiLabelClassificationWithBatchNormalization(X_train_mlb, X_test_mlb, y_train_mlb, y_test_mlb)
#plotTrainAccuracyAndLoss(modelHistoryMLC)



# case importance - multi class classification
# one_hot_encoded = pd.get_dummies(df['IMPORTANCE'], prefix='importance')
# one_hot_encoded = one_hot_encoded.astype(int)
#
# X = vect.fit_transform(df.TEXT)
# y = one_hot_encoded.values
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# modelHistoryMLC = runMultiClassification(X_train, X_test, y_train, y_test)
# plotTrainAccuracyAndLoss(modelHistoryMLC)