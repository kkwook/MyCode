from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pprint import pprint as pp
df = pd.read_csv("heights.csv")
df.head()

X = df['height']
X
Y = df['weight']
Y

print(np.ndim(X))
print(np.shape(X))
pp(np.ndim(Y))
pp(np.shape(Y))

plt.plot(X, Y, 'o')
line_fitter = LinearRegression()
X = X.values.reshape(-1, 1)
print(np.ndim(X))
print(np.shape(X))
line_fitter.fit(X, Y)

pp(np.ndim([[70]]))
pp(np.shape([[70]]))

line_fitter.predict([[70]])
line_fitter.coef_
line_fitter.intercept_
plt.plot(X, Y, 'o')
plt.plot(X, line_fitter.predict(X))
plt.show()



''' 다중 선형 회귀 '''
df = pd.read_csv("manhattan.csv")
df.head()

x = df[['bedrooms', 'bathrooms', 'size_sqft', 'min_to_subway', 'floor', 'building_age_yrs', 'no_fee',
        'has_roofdeck', 'has_washer_dryer', 'has_doorman', 'has_elevator', 'has_dishwasher', 'has_patio', 'has_gym']]
y = df[['rent']]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, test_size=0.2)
x_train
mlr = LinearRegression()
mlr.fit(x_train, y_train)

my_apartment = [[1, 1, 620, 16, 1, 98, 1, 0, 1, 0, 0, 1, 1, 0]]
my_predict = mlr.predict(my_apartment)
my_predict

y_predict = mlr.predict(x_test)
y_predict
plt.scatter(y_test, y_predict, alpha=0.4)
plt.xlabel("Actual Rent")
plt.ylabel('Predicted Rent')
plt.title("MULTIPLE LINEAR REGRESSION")
# 주택의 면적
plt.scatter(df[['size_sqft']], df[['rent']], alpha=0.4)
# 주택이 얼마나 오래지어졌는지
plt.scatter(df[['building_age_yrs']], df[['rent']], alpha=0.4)
np.shape(y)
np.shape(y_predict)
print(mlr.score(x_train, y_train))



''' 허쌤 책 코드 148부터 '''
from scipy import stats
X = [32, 64, 96, 118, 126, 144, 152, 158]
Y = [17, 24, 62, 49, 52, 105, 130, 125]
slope, intercept, r_value, p_value, std_err = stats.linregress(X, Y)
slope
intercept
r_value
p_value
std_err

''' p181 연습문제 '''
import statsmodels.api as sm
cars = sm.datasets.get_rdataset("cars", package = "datasets")
cars_df = cars.data
cars_df.columns

from sklearn.linear_model import LinearRegression
model = sm.OLS.from_formula("dist ~ speed", data = cars_df)
result = model.fit()
result.summary()
''' Durbin-Watson 검정값이 2와 가까우면 자기상관 x '''
result.rsquared
result.pvalues
result.tvalues
''' scipy stats 이용한 방식 '''
slope, intercept, r_value, p_value, std_err = stats.linregress(cars_df['speed'], cars_df['dist'])
slope
intercept
r_value
p_value
std_err

# 실제 값과 예측 값 산점도
%matplotlib inline
fig = plt.figure()
plt.scatter(cars_df['dist'], y_pred, c = "r", marker="o")
plt.xlabel("Target Y")
plt.ylabel("Predicted Y")
plt.show()

# P-P 도표
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline
y_pred = result.predict(); y_pred
plt.figure(figsize=(7, 7))
stats.probplot(y_pred, plot=plt)
plt.title("Probability plot")
plt.show()

# 레버리지와 s_resid
influence = result.get_influence()
leverage = influence.hat_matrix_diag; leverage # 레버리지 = 이상치 탐색
inf_df = influence.summary_frame(); inf_df.head()

''' 레버리지와 표준화 잔차 이용한 산점도 '''
plt.scatter(influence.hat_matrix_diag, inf_df.standard_resid)
fit = np.polyfit(influence.hat_matrix_diag, inf_df.standard_resid, 1)
fit_fn = np.poly1d(fit)
plt.plot(influence.hat_matrix_diag, fit_fn(influence.hat_matrix_diag), "r")
plt.xlabel("Leverage")
plt.ylabel("standard_resid")
plt.title("L vs SR")
plt.show()








W



