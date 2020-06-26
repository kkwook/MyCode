''' fitting and predicting '''
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(random_state=0)
X = [[ 1,  2,  3],  # 2 samples, 3 features
     [11, 12, 13]]
y = [0, 1]  # classes of each sample
clf.fit(X, y)
clf.predict(X)  # predict classes of the training data
clf.predict([[4, 5, 6], [14, 15, 16]])  # predict classes of new data
clf.predict([[4, 5, 200]])  # predict classes of new data
''' Transformers and pre_processors '''
from sklearn.preprocessing import StandardScaler # 평균이 0과 표준편차가 1이 되도록 반환
X = [[0, 15], [1, -10]]
StandardScaler().fit(X).transform(X)
''' Pipeline : chaning pre-processors 전처리 and estimators 추정 '''
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#pipeline object 파이프라인을 이용해 전처리객체와 분류 모형을 결합
pipe = make_pipeline(
    StandardScaler(),
    LogisticRegression(random_state=0)  # random_state를 0으로 두면 난수를 고정
)
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
# fit the whole pipeline
pipe.fit(X_train, y_train)
accuracy_score(pipe.predict(X_test), y_test)

''' Model evaluation '''
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate

X, y = make_regression(n_samples = 1000, random_state = 0)
lr = LinearRegression()
result = cross_validate(lr, X, y); result
result['test_score']

''' SVM 시각화 '''
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs


# we create 40 separable points
X, y = make_blobs(n_samples=40, centers=2, random_state=6)

# fit the model, don't regularize for illustration purposes
clf = svm.SVC(kernel='linear', C=1000)
clf.fit(X, y)

plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)

# plot the decision function
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)

# plot decision boundary and margins
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])
# plot support vectors
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
           linewidth=1, facecolors='none', edgecolors='k')
plt.show()

''' SVM 연습 '''
''' Classification ''' 
from sklearn import svm
X = [[0, 0], [1, 1]]
y = [0, 1]
clf = svm.SVC()
clf.fit(X,y)
clf.predict([[2, 2]])
# get support vectors
clf.support_vectors_
# get indices of sv
clf.support_
# get number of sv for each class
clf.n_support_

''' Multi-Class Classification '''
X = [[0], [1], [2], [3]]
Y = [0, 1, 2, 3]
clf = svm.SVC(decision_function_shape='ovo') 
# ovo => one-vs-one, K개의 타겟 클래스가 존재하는 경우, 이 중 2개의 클래스 조합을 선택하여 
# K(K-1)/2개의 이진 클래스 분류 문제를 풀고 이진판별을 통해 가장 많은 판별값을 얻은 클래스를 선택 
# ovr => one-vs-rest, K개의 클래스가 존재하는 경우, 각각의 클래스에 대해 
# 표본이 속하는 지 속하지 않는 지의 이진 문제를 푼다. ovo와 달리 K개 클래스 수 만큼만 풀면됨, 현재 가장 많이 사용됨
clf.fit(X,Y)
dec = clf.decision_function([[1]])
dec.shape[1] # 4 classes => 4*3/2 = 6
clf.decision_function_shape = "ovr"
dec = clf.decision_function([[1]])
dec.shape[1] # 4 classes

lin_clf = svm.LinearSVC()
lin_clf.fit(X, Y)
dec = lin_clf.decision_function([[1]])
dec.shape[1]










