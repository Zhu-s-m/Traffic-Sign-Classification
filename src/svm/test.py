from sklearn import decomposition, svm
from sklearn.model_selection import cross_val_score, ShuffleSplit
from src.data import Data
import time


def main():
    data = Data()
    x_train, y_train = data.get_data(data.CROP_DIR)
    pca = decomposition.IncrementalPCA()
    trainW = pca.fit_transform(x_train)  # fit the training set
    shufspl = ShuffleSplit(train_size=.7, test_size=.2, n_splits=5)
    svmclf = svm.SVC(kernel='rbf',verbose=1)
    scores = cross_val_score(svmclf, trainW, y_train, cv=shufspl)
    print("cross valid kernel pca + svm =", scores)


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print('Running time: %s Seconds' % (end - start))
