from flask import Flask, request, jsonify

from ClassifierHelper.ClassifierTrainer import ClassifierTrainer
from ClassifierHelper.DirectoryHelper import DirectoryHelper

# Flask implementation
app = Flask(__name__)

@app.route('/api/Classifier', methods=['GET'])
def hello():
    return jsonify({'Message': 'Hello world!!'})

@app.route('/api/classifier/Train', methods=['POST'])
def train():
    directory_helper = DirectoryHelper("./Datasets")
    datasets = directory_helper.get_training_dataset()
    classifier_trainer = ClassifierTrainer()
    data = request.get_json()
    percentage = data['train_percentage']
    classifier_trainer.train(percentage, datasets)
    return jsonify({'Result': 'Ok'})

@app.route('/api/classifier/classify', methods=['POST'])
def classify():
    classifier_trainer = ClassifierTrainer()
    data = request.get_json()
    document = data['doc']
    (result, confidence) = classifier_trainer.classify(document)
    return jsonify({'Result': result, 'Confidence': confidence})


if __name__ == '__main__':
    app.run(debug=True)

