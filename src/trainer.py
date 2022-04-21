import os
import pickle
import sys

import numpy
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from src.data import Data


def main():
    data = Data()
    # train_dir=input("Please enter the train data")
    train_dir = data.ORIGINAL_DIR
    print('Loading Data')
    train_x, train_y = data.load_data(train_dir)
    print('--------------------')

    nn = MLPClassifier(hidden_layer_sizes=(100, 20))
    svm = SVC(kernel='rbf', C=20,decision_function_shape='ovo')
    rf = RandomForestClassifier(n_estimators=150, n_jobs=-1, random_state=100)

    print('Neural Network Model Training')
    nn.fit(train_x, train_y)
    print('--------------------')

    print('Random Forest Model Training')
    rf.fit(train_x, train_y)
    print('--------------------')

    print('SVM Model Training')
    # Process specific classes
    train_x, train_y, svm_five = svm_train(train_x, train_y, 5)
    # Process common classes
    svm.fit(train_x, train_y)
    print('--------------------')

    print('Saving Models')
    with open(os.path.join(data.MODEL_DIR, 'nn.model'), "wb") as f:
        pickle.dump(nn, f)

    with open(os.path.join(data.MODEL_DIR, 'svm.model'), "wb") as f:
        pickle.dump(svm, f)

    with open(os.path.join(data.MODEL_DIR, 'svm_five.model'), "wb") as f:
        pickle.dump(svm_five, f)

    with open(os.path.join(data.MODEL_DIR, 'rf.model'), "wb") as f:
        pickle.dump(rf, f)

    print("Models have been saved at " + data.MODEL_DIR)
    print(os.listdir(data.MODEL_DIR))
    print('Please do not change the name and location of each file')


def svm_train(x, y, target_label):
    y_target = []

    for i in range(len(y)):
        # Binary classification of specific classes
        if y[i] == target_label:
            y_target.append(1)
        else:
            y_target.append(0)
    svm_target = SVC(kernel='rbf', C=10,decision_function_shape='ovo')
    svm_target.fit(x, y_target)
    for i in range(len(y)):
        if y_target == 1:
            x = numpy.delete(x, i)
            y = numpy.delete(y, i)
    return x, y, svm_target


if __name__ == '__main__':
    main()
