import os
import pickle

from sklearn.metrics import classification_report
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.svm import SVC
from src.data import Data
import time
import numpy

data=Data()
x,y=data.get_data(data.CROP_DIR,by_file=True)
def train_test(x, y):
    clf = SVC(verbose=1,kernel='rbf',C=20,decision_function_shape='ovo')
    #shuffle = ShuffleSplit(train_size=.7, test_size=.2, n_splits=5)
    scores = cross_val_score(clf, x, y, cv=10)
    print("Cross validation scores:{}".format(scores))
    print("Mean cross validation score:{:2f}".format(scores.mean()))
    print("Standard error validation score:{:2f}".format(scores.std()))
    print("Finish training")



def test():
    data = Data()
    x_test_crop,y_test_crop=data.get_data(data.TEST_CROP_DIR,by_file=True)
    print("Start training")
    rfc = SVC(verbose=1,kernel='rbf',C=20,decision_function_shape='ovo')
    target_names = ['class 1', 'class 2', 'class 3', 'class 4', 'class 5',
                    'class 6', 'class 7', 'class 8', 'class 9', 'class 10',
                    'class 11', 'class 12', 'class 13', 'class 14', 'class 15']

    x, y = data.get_data(data.TRAIN_DIR,by_file=True)
    rfc.fit(x,y)
    print(len(x))
    y_pred_crop=rfc.predict(x_test_crop)
    print(classification_report(y_test_crop, y_pred_crop, target_names=target_names))

# 6

def generate_model(x,y):
    print("Start training")
    rfc = rfc = SVC(C=20, verbose=1)

    rfc.fit(x,y)
    with open(os.path.join(data.MODEL_DIR, 'svm.model'),"wb") as f:
        pickle.dump(rfc, f)
if __name__ == "__main__":
    start = time.time()
    test()
    #train_test(x,y)
    #generate_model(x,y)
    end = time.time()
    print('Running time: %s Seconds' % (end - start))
