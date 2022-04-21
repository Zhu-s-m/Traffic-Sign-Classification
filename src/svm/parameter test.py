
import numpy as np
from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.svm import SVC
from src.data import Data


data = Data()
step=10
min_estimators = 1
max_estimators = 100
train_scores = []
test_scores = []
x,y=data.get_data(data.CROP_DIR,by_file=True)

clf = SVC(C=1.0, verbose=1,decision_function_shape='ovo')
for i in range(min_estimators, max_estimators + 1, step):
        clf.set_params(C=i,kernel='rbf')
        # clf.fit(x,y)
        train_scores.append(np.mean(cross_val_score(clf, x, y, cv=3)))
        # test_scores.append(clf.score(X_test, y_test))
        # train_scores.append(0)

fig, ax = plt.subplots(dpi = 100)
ax.set_xlabel("estimators")
ax.set_ylabel("accuracy")
# ax.set_title("Accuracy vs estimators for training and testing sets")
# print(len(train_scores))
# print(len(range(min_estimators, max_estimators + 1, 10)))
ax.plot(range(min_estimators, max_estimators + 1, step), train_scores, label="train",
        drawstyle="steps-post")
#ax.plot(range(min_estimators, max_estimators + 1, 5), test_scores, label="test",
#        drawstyle="steps-post")
ax.legend()
plt.show()
print("Finish")