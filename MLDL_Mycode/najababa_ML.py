''' ML 모형평가 '''
# 73
import numpy as np

#74
table = np.array([[1613, 22],
                  [81, 77]])

#75
def fmeasure(table):
    precision = table[1, 1] / (table[0, 1] + table[1, 1])  # TP / (FP + TP)
    recall = table[1, 1] / (table[1, 0] + table[1, 1])  # TP / (FN + TP)
    return (2 * precision * recall / (precision + recall))


round(fmeasure(table), 5)

''' 회귀분석 '''

''' 분류분석 '''
# 240
y_true = [1, 1, 0, 0, 2, 1, 0, 2, 2]
len(y_true)
y_pred = [1, 1, 0, 1, 1, 0, 0, 2, 1]

from sklearn.metrics import confusion_matrix
confusion_matrix(y_true, y_pred)
# 241
import pandas as pd
df = pd.DataFrame({"y_true":y_true, "y_pred":y_pred})
pd.crosstab(df.y_true, df.y_pred, margins=True)
import platform; print(platform.platform())
import sys; print("Python", sys.version)
import numpy; print("NumPy", numpy.__version__)
import scipy; print("SciPy", scipy.__version__)
import sklearn; print("Scikit-Learn", sklearn.__version__)
from sklearn.metrics import jaccard_score
import pandas_ml as pdml
from sklearn.metrics import jaccard_similarity_score
Confusion_matriX = confusion_matrix(y_true, y_pred)
Confusion_matriX

%matplotlib inline
import matplotlib.pyplot as plt
plt.plot(Confusion_matriX)
plt.show()

result = pd.read_csv("http://javaspecialist.co.kr/pds/382")
result.head()
from sklearn.metrics import accuracy_score
accuracy_score(result.y_true, result.y_pred)

# p.244
from sklearn.metrics import f1_score
f1_score(result.y_true, result.y_pred)

# p.245
import numpy as np
confusion_matrix = np.array([[15,  0,  0],
                             [ 0, 13,  3],
                             [ 0,  0, 14]])

from sklearn.metrics import classification_report
print(classification_report(test_y, pred, 
                     target_names=["setosa", "versicolor", "virginica"]))
                     