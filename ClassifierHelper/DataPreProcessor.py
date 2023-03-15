# Data Preprocessor will clean the data and tokenize the sentence provide the valid training data.
#
import nltk
from nltk import word_tokenize


class DataPreProcessor:
    def __init__(self, datasets):
        self._datasets = datasets
        self._documents = []
        self._workflow_words = {}
        print("Preprocessing the data for training")

    def generate_documents(self):

        for key in self._datasets:
            workflow_name = key.split('.')[0]
            print("Cleaning up the data related to ", workflow_name)
            w_words = []
            for doc in self._datasets[key]:
                self._documents.append((doc, workflow_name))
                words = word_tokenize(doc)
                for w in words:
                    w_words.append(w.lower())
            self._workflow_words[workflow_name] = w_words

        return self._documents

    def get_word_features(self):
        word_features = []
        for key in self._workflow_words:
            all_words = nltk.FreqDist(self._workflow_words[key])
            word_features.extend(list(all_words.keys())[:100])

        return word_features

