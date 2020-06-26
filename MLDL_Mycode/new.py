''' XGBoost 예측모형 자동화 '
    https://statkclee.github.io/model/model-python-xgboost-hyper.html '''

from sklearn.model_selection import KFold, GridSearchCV
import xgboost as xgb

xgb_clf = xgb.XGBClassifier()
print(xgb_clf)

param_grid={'booster' :['gbtree'],
                 'silent':[True],
                 'max_depth':[5,6,8],
                 'min_child_weight':[1,3,5
                 'gamma':[0,1,2,3],
                 'nthread':[4],
                 'colsample_bytree':[0.5,0.8],
                 'colsample_bylevel':[0.9],
                 'n_estimators':[50],
                 'objective':['binary:logistic'],
                 'random_state':[2]}

# 3번
cv=KFold(n_splits=6, random_state=1)

# 4번
gcv=GridSearchCV(model, param_grid=param_grid, cv=cv, scoring='f1', n_jobs=4)

# 5번
gcv.fit(train_X.values,train_Y.values)
print('final params', gcv.best_params_)   # 최적의 파라미터 값 출력
print('best score', gcv.best_score_)      # 최고의 점수

final params {'booster': 'gbtree', 'colsample_bylevel': 0.9, 'colsample_bytree': 0.8, 'gamma': 0, 'max_depth': 8, 'min_child_weight': 3, 'n_estimators': 50, 'nthread': 4, 'objective': 'binary:logistic', 'random_state': 2, 'silent': True}
best score 0.7618358747302295

from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler


def rfr_model(X, y):
    # Perform Grid-Search
    gsc = GridSearchCV(
        estimator=RandomForestClassifier(),
        param_grid={
            'max_depth': range(3, 7),
            'n_estimators': (10, 50, 100, 1000)}, cv=5, scoring='neg_mean_squared_error', verpose=0, n_jobs=-1)

    grid_result = gsc.fit(X, y)
    best_params = grid_result.best_params_
    rfr = RandomForestRegressor(max_depth=best_params['max_depth'],
                                n_estimators=best_params['n_estimators'],
                                random_state=False, verbose=False) 
    # n_jobs=-1로 지정해주면 모든 코어를 다 사용하기때문에 컴퓨터는 뜨거워지겠지만, 속도는 많이 빨라집니다. 
    #  verbose로 log 출력의 level을 조정

    # Perform K-Fold CV
    scores = cross_val_score(
        rfr, X, y, cv=10, scoring="neg_mean_squared_error")

    return scores

scores = cross_val_score(rfr, X, y, cv=10, scoring='neg_mean_absolute_error')
predictions = cross_val_predict(rfr, X, y, cv=10)


