''' 허교수님 책 2장 p.72부터 '''
import seaborn as sns
iris = sns.load_dataset("iris")

X = iris.iloc[:,:-1]; X
y = iris.species; y
# from sklearn.model_selection import train_test_split
# train_X, test_X, train_y, test_y = train_test_split(X, y, stratify = y, test_size = 0.3) # stratify : 계층적 데이터 추출 옵션
# test_y.value_counts()
# from sklearn.ensemble import RandomForestClassifier
# rf_model = RandomForestClassifier()
# rf_model.fit(train_X, train_y)

''' 랜덤 포레스트 예시 ''' 
import pandas as pd
from sklearn import datasets
if __name__ == "__main__":
    iris = datasets.load_iris()
    print("아이리스 종류 :", iris.target_names)
    print("target : [0:setosa, 1:versicolor, 2:virginica]")
    print("데이터 수 : ", len(iris.data))
    print("데이터 열 이름 : ", iris.feature_names)
    
    data = pd.DataFrame(
        {
            "sepal length" : iris.data[:, 0],
            "sepal width" : iris.data[:, 1],
            "petal length" : iris.data[:, 2],
            "petal width" : iris.data[:, 3],
            'species' : iris.target
        }
    )
    print(data.head())

from sklearn.model_selection import train_test_split
x = data[['sepal length', 'sepal width', 'petal length', 'petal width']]
y = data['species']
 
# 테스트 데이터 30%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
print(len(x_train))
print(len(x_test))
print(len(y_train))
print(len(y_test))
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
forest = RandomForestClassifier(n_estimators = 100)
forest.fit(x_train, y_train)
y_pred = forest.predict(x_test)
print(y_pred); print(list(y_test))
print("정확도 : ", metrics.accuracy_score(y_test, y_pred))

# from sklearn.neural_network import MLPClassifier
# mlp_model = MLPClassifier(hidden_layer_sizes = (50, 30))
# mlp_model.fit(train_X, train_y)
# pred = mlp_model.predict(test_X); pred
