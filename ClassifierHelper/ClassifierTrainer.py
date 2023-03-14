# Classifier Trainer to train the model with training dataset and store the classifier brain in pickle.
import random
import pickle
import nltk
from nltk import word_tokenize
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

from ClassifierHelper.DataPreProcessor import DataPreProcessor
from ClassifierHelper.VoteClassifier import VoteClassifier


class ClassifierTrainer:
    def __init__(self):
        print("Classifier training beginning..")

    def train(self, percentage, datasets):
        data_pre_processor = DataPreProcessor(datasets)
        # Generate the document with pair eg ( workflow_name, sentence)
        documents = data_pre_processor.generate_documents()
        # Generate word features based on Freq distribution
        word_features = data_pre_processor.get_word_features()

        word_feature_f = open("./Pickle/WordFeature/word_feature.pickle", "wb")
        pickle.dump(word_features, word_feature_f)
        word_feature_f.close()

        # Create a feature sets to train and test the model.
        feature_sets = [(self.find_features(rev, word_features), category) for (rev, category) in documents]
        random.shuffle(feature_sets)
        # divide training and testing dataset.
        split_index = int(len(feature_sets) * percentage)
        print("Dividing training and testing dataset based on percentage ", percentage * 100)
        training_set = feature_sets[:split_index]
        testing_set = feature_sets[split_index:]
        print("Split index ", split_index)

        naivebayes_f = open("./Pickle/Classifiers/NaiveBayes.pickle", "wb")
        classifier = nltk.NaiveBayesClassifier.train(training_set)
        print("Original Naive Bayes Algo Classifier ", (nltk.classify.accuracy(classifier, testing_set)) * 100)
        pickle.dump(classifier, naivebayes_f)
        naivebayes_f.close()

        mnb_classifier_f = open("./Pickle/Classifiers/MnbClassifier.pickle", "wb")
        MNB_Classifier = SklearnClassifier(MultinomialNB())
        MNB_Classifier.train(training_set)
        print("MNB_Classifier Naive Bayes Algo Classifier ", (nltk.classify.accuracy(MNB_Classifier, testing_set)) * 100)
        pickle.dump(MNB_Classifier, mnb_classifier_f)
        mnb_classifier_f.close()


        bernoulli_classifier_f = open("./Pickle/Classifiers/BernoulliClassifier.pickle", "wb")
        BernoulliNB_Classifier = SklearnClassifier(BernoulliNB())
        BernoulliNB_Classifier.train(training_set)
        print("Bernoulli_NB Naive Bayes Algo Classifier ", (nltk.classify.accuracy(BernoulliNB_Classifier, testing_set)) * 100)
        pickle.dump(BernoulliNB_Classifier, bernoulli_classifier_f)
        bernoulli_classifier_f.close()

        logistic_classifier_f = open("./Pickle/Classifiers/LogisticReg.pickle", "wb")
        LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
        LogisticRegression_classifier.train(training_set)
        print("LogisticRegression Naive Bayes Algo Classifier ", (nltk.classify.accuracy(LogisticRegression_classifier,
                                                                                         testing_set)) * 100)
        pickle.dump(LogisticRegression_classifier, logistic_classifier_f)
        logistic_classifier_f.close()

        # SGDClassifier
        sgd_classifier_f = open("./Pickle/Classifiers/SGDClassifier.pickle", "wb")
        SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
        SGDClassifier_classifier.train(training_set)
        print("SGDClassifier Algo Classifier ", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set)) * 100)
        pickle.dump(SGDClassifier_classifier, sgd_classifier_f)
        sgd_classifier_f.close()

        # SVC
        svc_classifier_f = open("./Pickle/Classifiers/SVCClassifier.pickle", "wb")
        SVCClassifier_classifier = SklearnClassifier(SVC())
        SVCClassifier_classifier.train(training_set)
        print("SVC Algo Classifier ", (nltk.classify.accuracy(SVCClassifier_classifier, testing_set)) * 100)
        pickle.dump(SVCClassifier_classifier,svc_classifier_f)
        svc_classifier_f.close()

        # LinearSVC
        linear_svc_f = open("./Pickle/Classifiers/LinearSVC.pickle", "wb")
        LinearSVC_classifier = SklearnClassifier(LinearSVC())
        LinearSVC_classifier.train(training_set)
        print("SVC Algo Classifier ", (nltk.classify.accuracy(LinearSVC_classifier, testing_set)) * 100)
        pickle.dump(LinearSVC_classifier, linear_svc_f)
        linear_svc_f.close()

        # NuSvC
        NuSVC_classifier_f = open("./Pickle/Classifiers/NuSVC.pickle", "wb")
        NuSVC_Classifier = SklearnClassifier(NuSVC())
        NuSVC_Classifier.train(training_set)
        print("Nu SVC Classifier ", (nltk.classify.accuracy(NuSVC_Classifier, testing_set)) * 100)
        pickle.dump(NuSVC_Classifier, NuSVC_classifier_f)
        NuSVC_classifier_f.close()

    def classify(self, document):
        word_features_f = open("./Pickle/WordFeature/word_feature.pickle", "rb")
        word_features = pickle.load(word_features_f)
        word_features_f.close()

        naivebayes_f = open("./Pickle/Classifiers/NaiveBayes.pickle", "rb")
        navibayes_classifier = pickle.load(naivebayes_f)
        naivebayes_f.close()

        mnb_classifier_f = open("./Pickle/Classifiers/MnbClassifier.pickle", "rb")
        mnb_classifier = pickle.load(mnb_classifier_f)
        mnb_classifier_f.close()

        bernoulli_classifier_f = open("./Pickle/Classifiers/BernoulliClassifier.pickle", "rb")
        bernoulli_classifier = pickle.load(bernoulli_classifier_f)
        bernoulli_classifier_f.close()

        logistic_classifier_f = open("./Pickle/Classifiers/LogisticReg.pickle", "rb")
        logistic_classifier = pickle.load(logistic_classifier_f)
        logistic_classifier_f.close()

        sgd_classifier_f = open("./Pickle/Classifiers/SGDClassifier.pickle", "rb")
        sgd_classifier = pickle.load(sgd_classifier_f)
        sgd_classifier_f.close()

        svc_classifier_f = open("./Pickle/Classifiers/SVCClassifier.pickle", "rb")
        svc_classifier = pickle.load(svc_classifier_f)
        svc_classifier_f.close()

        linear_svc_f = open("./Pickle/Classifiers/LinearSVC.pickle", "rb")
        linear_svc_classifier = pickle.load(linear_svc_f)
        linear_svc_f.close()

        nusvc_classifier_f = open("./Pickle/Classifiers/NuSVC.pickle", "rb")
        nusvc_classifier = pickle.load(nusvc_classifier_f)
        nusvc_classifier_f.close()


        voted_classifier = VoteClassifier(navibayes_classifier,mnb_classifier,bernoulli_classifier,logistic_classifier,
                                          sgd_classifier,svc_classifier,linear_svc_classifier,nusvc_classifier)

        feature_sets = self.find_features(document, word_features)

        classification_result = voted_classifier.classify(feature_sets)

        confidence = voted_classifier.confidence(feature_sets)

        return (classification_result, confidence)


    def find_features(self, document, word_features):
        words = word_tokenize(document)
        features = {}
        for w in word_features:
            features[w] = (w in words)
        return features
