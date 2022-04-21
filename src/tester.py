import os
import pickle
import sys

import numpy
import pandas
from skimage import transform
from PIL import Image
from skimage.color import rgb2gray
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from src.data import Data

def main():
    data = Data()
    # test_dir=input("Plaese enter the test data path")

    print('Loading Test Data')
    test_x,test_y=data.load_data(data.ORIGINAL_DIR)
    print('--------------------')

    print('Loading Neural Network Model File')
    file=open(os.path.join(data.MODEL_DIR,'nn.model'), 'rb')
    nn=pickle.load(file)
    print('--------------------')

    print('Loading Random Forest Model File')
    file = open(os.path.join(data.MODEL_DIR, 'rf.model'), 'rb')
    rf = pickle.load(file)
    print('--------------------')

    print('Loading SVM Model File')
    file = open(os.path.join(data.MODEL_DIR, 'svm.model'), 'rb')
    svm = pickle.load(file)
    file = open(os.path.join(data.MODEL_DIR, 'svm_five.model'), 'rb')
    svm_five = pickle.load(file)
    print('--------------------')

    models=[nn,rf,svm]
    name=['Neural Network','RandomForest','SVM']

    target_names = ['class 0','class 1', 'class 2', 'class 3', 'class 4',
                    'class 5','class 6', 'class 7', 'class 8', 'class 9',
                    'class 10','class 11', 'class 12', 'class 13', 'class 14']

    for model in range(len(models)-1):
        print(name[model]+" Result Report")
        pred_y=models[model].predict(test_x)
        report=classification_report(test_y, pred_y, target_names=target_names)
        print(report)
    print("SVM Result Report")
    pred_y = []
    for item in test_x:
        temp = []
        temp.append(item)
        temp = numpy.array(temp)
        if svm_five.predict(temp) == [1]:
            pred_y.append(5)
        else:
            pred_y.append(svm.predict(temp)[0])
    target_names = ['class 1', 'class 2', 'class 3', 'class 4', 'class 5',
                    'class 6', 'class 7', 'class 8', 'class 9', 'class 10',
                    'class 11', 'class 12', 'class 13', 'class 14', 'class 15']
    print(classification_report(test_y, pred_y, target_names=target_names))


if __name__ == "__main__":
    main()
