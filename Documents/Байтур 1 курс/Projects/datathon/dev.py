import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

SEED = 1
random.seed(SEED)
np.random.seed(SEED)

train = pd.read_csv('file_train.csv').sort_values(by=['PoS_word', 'Tag', 'Affix'])
test = pd.read_csv('file_test.csv').sort_values(by=['PoS_word', 'Tag', 'Affix'])
df = pd.concat([train, test], ignore_index=True)

X = df[['Word', 'Root', 'Affix', 'PoS_root', 'PoS_word']]
y = df['Tag']

X_pr = pd.get_dummies(X)
le = LabelEncoder()
y = le.fit_transform(y)

train_X = X_pr.iloc[:train.shape[0]]
train_y = y[:train.shape[0]]
train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, test_size=0.05, random_state=SEED)

rf = RandomForestClassifier(n_estimators=100, random_state=SEED)
rf.fit(train_X, train_y)

rf_predict_result = rf.predict(val_X)

f1_micro = f1_score(val_y, rf_predict_result, average='micro')
print("F1 score:", f1_micro)

test_X = X_pr.iloc[train.shape[0]:]
predictions = rf.predict(test_X)

test['Tag'] = le.inverse_transform(predictions)
test[['Word', 'Root', 'Affix', 'Tag']].to_csv('my_submission2.csv', index=False, header=True)