import time
import pickle
import os
from sklearn import neural_network
from sklearn.metrics import classification_report, accuracy_score, make_scorer
from sklearn.model_selection import ShuffleSplit, cross_val_score

from src.data import Data


def train_test(x, y):
    nn = neural_network.MLPClassifier(verbose=1)
    nn.fit(x,y)
    print("Finish training")
    return nn

# whalambda hhhh什麼寫不出去模型写不出去你是說分類器導不出去
# 确实
# Traceback (most recent call last):
#   File "C:\Box\MachineLearningProject\src\nn\test.py", line 27, in <module>
#     pickle.dump(nn, os.path.join(data.MODEL_DIR, 'nn.model'))
# TypeError: file must have a 'write' attribute


if __name__ == "__main__":
    start = time.time()
    # test()
    data = Data()
    print(os.path.join(data.MODEL_DIR, 'nn.model'))
    x, y = data.get_data(data.CROP_DIR,by_file=True)
    nn=train_test(x, y)
    with open(os.path.join(data.MODEL_DIR, 'nn.model'),"wb") as f:
        pickle.dump(nn, f)
    end = time.time()
    print('Running time: %s Seconds' % (end - start))