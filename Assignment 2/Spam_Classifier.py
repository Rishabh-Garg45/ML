import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

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

data = pd.read_csv('final_data.csv')
data.fillna('', inplace=True)

vectorizer = CountVectorizer(binary=True)
X = vectorizer.fit_transform(data['email'])
y = data['label'].values

nb_classifier = ScratchNaiveBayesClassifier()
nb_classifier.fit(X, y)

test_files = ['test_sample_1.csv', 'test_sample_2.csv', 'test_sample_3.csv', 'test_sample_4.csv']
y_tests = []
y_preds = []
cm_list = []

for test_file in test_files:

    test_data = pd.read_csv(test_file)
    X_test = test_data.email
    y_test = test_data.label
    
    X_test_vectorized = vectorizer.transform(X_test)
    y_pred = nb_classifier.predict(X_test_vectorized)
    
    print(classification_report(y_test, y_pred))
    
    y_tests.append(y_test)
    y_preds.append(y_pred)
    cm_list.append(confusion_matrix(y_test, y_pred))

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for i, ax in enumerate(axes.flat):
    sns.heatmap(cm_list[i], annot=True, fmt='d', ax=ax, cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    ax.set_title(f'Confusion Matrix for Test Sample {i + 1}')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')

plt.tight_layout()
plt.show()