import numpy as np
import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer

train_data = pd.read_csv("final_data.csv")
train_data.fillna('', inplace=True)

vectorizer = CountVectorizer(binary=True)
X = vectorizer.fit_transform(train_data['email'])
y = train_data['label'].values

class ScratchNaiveBayesClassifier:
    
    def __init__(self):
        self.spam_word_probs = None
        self.ham_word_probs = None
        self.spam_prior = 0
        self.ham_prior = 0
        self.vocab_size = 0

    def fit(self, X, y):
        
        spam_count = np.sum(y == 1)
        ham_count = np.sum(y == 0)
        self.spam_prior = spam_count / len(y)
        self.ham_prior = ham_count / len(y)

        spam_word_count = X.T @ (y == 1)
        ham_word_count = X.T @ (y == 0)

        self.vocab_size = X.shape[1]
        spam_word_count += 1  
        ham_word_count += 1  

        total_spam_words = spam_word_count.sum() + self.vocab_size  
        total_ham_words = ham_word_count.sum() + self.vocab_size  

        self.spam_word_probs = spam_word_count / total_spam_words
        self.ham_word_probs = ham_word_count / total_ham_words

    def predict(self, X):
        
        log_spam_likelihood = np.log(self.spam_prior) + X @ np.log(self.spam_word_probs)
        log_ham_likelihood = np.log(self.ham_prior) + X @ np.log(self.ham_word_probs)

        return (log_spam_likelihood > log_ham_likelihood).astype(int)

nb_classifier = ScratchNaiveBayesClassifier()
nb_classifier.fit(X, y)

def make_predictions():
    test_emails = []
    for f in os.listdir("test"):
        if f.endswith(".txt"):
            file_path = os.path.join("test", f)
            with open(file_path, 'r', encoding='utf-8') as email:
                c = email.read().strip()
            test_emails.append({'email_name': f, 'email': c})
    test = pd.DataFrame(test_emails)
    test.fillna('', inplace = True)
    X_test = vectorizer.transform(test.email)
    y_pred = nb_classifier.predict(X_test)
    return y_pred

pred = make_predictions()
print(pred)

predictions_df = pd.DataFrame({'predictions': pred})
predictions_df.to_csv('predictions.csv', index=False)