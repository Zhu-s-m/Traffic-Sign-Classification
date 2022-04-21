
import numpy as np
from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from src.data import Data
import time

from sklearn import neural_network
from sklearn.metrics import classification_report, accuracy_score, make_scorer
from sklearn.model_selection import ShuffleSplit, cross_val_score

from src.data import Data

# data = Data()
# min_estimators = 1
# max_estimators = 300
# train_scores = []
# test_scores = []
#
# x, y = data.get_data(data.CROP_DIR)
# nn = neural_network.MLPClassifier(hidden_layer_sizes=(0, ), max_iter=10, alpha=1e-4,
#                     solver='sgd', verbose=10, random_state=1, learning_rate_init=.1)
# for i in range(min_estimators, max_estimators + 1, 20):
#         nn.set_params(hidden_layer_sizes=(i, ))
#         nnn=nn.fit(x, y)
#         train_scores.append(np.mean(cross_val_score(nnn, x, y, cv=3)))
#         #test_scores.append(randomForest.score(x, y))
#
# fig, ax = plt.subplots(dpi = 100)
# ax.set_xlabel("hide_layer")
# ax.set_ylabel("accuracy")
# ax.set_title("Accuracy vs estimators for training and testing sets")
# ax.plot(range(min_estimators, max_estimators + 1, 20), train_scores, label="train",
#         drawstyle="steps-post")
# #ax.plot(range(min_estimators, max_estimators + 1, 5), test_scores, label="test",
# #        drawstyle="steps-post")
# ax.legend()
# plt.show()
# print("Finish")
#
#



data = Data()
min_estimators = 50
max_estimators = 150
train_scores = []
test_scores = []

x, y = data.get_data(data.CROP_DIR,by_file=True)
nn = neural_network.MLPClassifier(hidden_layer_sizes=(100,0), max_iter=200, alpha=1e-4,
                    solver='adam', verbose=10, learning_rate_init=.001)
for i in range(min_estimators, max_estimators + 1, 10):
        nn.set_params(hidden_layer_sizes=(100,i))
        # nn=nn.fit(x, y)
        train_scores.append(np.mean(cross_val_score(nn, x, y, cv=3)))
        #test_scores.append(randomForest.score(x, y))

fig, ax = plt.subplots(dpi = 100)
ax.set_xlabel("hide_layer")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs estimators for training and testing sets")
ax.plot(range(min_estimators, max_estimators + 1, 10), train_scores, label="train",
        drawstyle="steps-post")
#ax.plot(range(min_estimators, max_estimators + 1, 5), test_scores, label="test",
#        drawstyle="steps-post")
ax.legend()
plt.show()
print("Finish")
