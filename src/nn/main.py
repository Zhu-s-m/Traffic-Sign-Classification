import os
import time

from sklearn import neural_network
from sklearn.metrics import classification_report, accuracy_score, make_scorer
from sklearn.model_selection import ShuffleSplit, cross_val_score
import pickle
from src.data import Data

# 默认参数
# hidden_layer_sizes=(100,),
#         activation="relu",
#         *,
#         solver="adam",
#         alpha=0.0001,
#         batch_size="auto",
#         learning_rate="constant",
#         learning_rate_init=0.001,
#         power_t=0.5,
#         max_iter=200,
#         shuffle=True,
#         random_state=None,
#         tol=1e-4,
#         verbose=False,
#         warm_start=False,
#         momentum=0.9,
#         nesterovs_momentum=True,
#         early_stopping=False,
#         validation_fraction=0.1,
#         beta_1=0.9,
#         beta_2=0.999,
#         epsilon=1e-8,
#         n_iter_no_change=10,
#         max_fun=15000,
def train_test(x, y):
    # nn = neural_network.MLPClassifier(verbose=1)
    nn = neural_network.MLPClassifier(hidden_layer_sizes=(100,20), verbose=1)
    # shuffle = ShuffleSplit(train_size=.7, test_size=.2, n_splits=5)
    scores = cross_val_score(nn, x, y, cv=10)
    print("Cross validation scores:{}".format(scores))
    print("Mean cross validation score:{:2f}".format(scores.mean()))
    print("Standard error validation score:{:2f}".format(scores.std()))
    print("Finish training")

# ??????????????/
# 参数太离谱了吧你点进去可以看他的默认参数只有那個solver函數不一樣
# 然后再比这那个改
def test():

    x_test_crop,y_test_crop=data.get_data(data.TEST_CROP_DIR,by_file=True)
    print("Start training")


    rfc = neural_network.MLPClassifier(verbose=1,hidden_layer_sizes=(100, 20))
    target_names = ['class 1', 'class 2', 'class 3', 'class 4', 'class 5',
                    'class 6', 'class 7', 'class 8', 'class 9', 'class 10',
                    'class 11', 'class 12', 'class 13', 'class 14', 'class 15']
    rfc.fit(x,y)
    y_pred_crop=rfc.predict(x_test_crop)
    print(classification_report(y_test_crop, y_pred_crop, target_names=target_names))

def classification_report_with_accuracy_score(y_true, y_pred):
    print(classification_report(y_true, y_pred))  # print classification report
    return accuracy_score(y_true, y_pred)  # return accuracy score

def generate_model(x,y):
    print("Start training")
    nn = neural_network.MLPClassifier(hidden_layer_sizes=(100,20),verbose=1)

    nn.fit(x,y)
    with open(os.path.join(data.MODEL_DIR, 'nn.model'),"wb") as f:
        pickle.dump(nn, f)

if __name__ == "__main__":
    start = time.time()
    data = Data()
    x, y = data.get_data(data.TRAIN_DIR,by_file=True)
    # rfc = neural_network.MLPClassifier(verbose=1)
    # nested_score = cross_val_score(rfc, X=x, y=y, cv=3, scoring=make_scorer(classification_report_with_accuracy_score))
    # print(nested_score)
    #train_test(x, y)
    # generate_model(x,y)
    test()
    end = time.time()
    print('Running time: %s Seconds' % (end - start))
