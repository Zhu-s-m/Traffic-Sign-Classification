import numpy
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from src.data import Data



def bin_train(x,y,target_label):
    y_target=[]

    for i in range(len(y)):
        if y[i]==target_label:
            y_target.append(1)
        else:
            y_target.append(0)
    svm_target=SVC(verbose=1,C=10)
    svm_target.fit(x,y_target)
    for i in range(len(y)):
        if y_target==1:
            x=numpy.delete(x,i)
            y=numpy.delete(y,i)
    return x,y,svm_target


def main():
    data = Data()
    x,y=data.get_data(data.TRAIN_DIR,by_file=True)
    x,y,svm_target=bin_train(x,y,5)
    svm=SVC(verbose=1,C=20,decision_function_shape='ovo')
    svm.fit(x,y)
    x_test, y_test=data.get_data(data.TEST_CROP_DIR,by_file=True)
    y_pred=[]
    for item in x_test:
        temp=[]
        temp.append(item)
        temp=numpy.array(temp)
        if svm_target.predict(temp)==[1]:
            y_pred.append(5)
        else:
            y_pred.append(svm.predict(temp)[0])
    target_names = ['class 1', 'class 2', 'class 3', 'class 4', 'class 5',
                    'class 6', 'class 7', 'class 8', 'class 9', 'class 10',
                    'class 11', 'class 12', 'class 13', 'class 14', 'class 15']
    print(classification_report(y_test, y_pred, target_names=target_names))


if __name__ == '__main__':
    main()



