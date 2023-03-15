from nltk.classify import ClassifierI
from statistics import mode


class VoteClassifier(ClassifierI):
    def labels(self):
        pass

    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def start_prediction(self, features):
        votes = []
        count = 0
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
            count = count + 1
        return votes

    def classify(self, feature_set):
        votes = self.start_prediction(feature_set)
        return mode(votes)

    def confidence(self, features):
        votes = self.start_prediction(features)
        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf
