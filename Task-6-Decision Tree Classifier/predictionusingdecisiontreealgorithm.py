import sklearn
from sklearn import datasets
import sklearn.datasets as datasets
import pandas as pd
import numpy as np
import seaborn as sns
print('Libraries Imported')
iris=datasets.load_iris()
X=pd.DataFrame(iris.data, columns=iris.feature_names)
print(X.head())
print(X.tail())
print('Dataset Info:{}'.format(X.info()))
print('Dataset Description:\n\n{}'.format(X.describe()))
X.nunique()
X.isnull().sum()
X.corr()
sns.heatmap(X.corr(), annot=True)
Y=iris.target 
print(Y)
from io import StringIO
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import metrics
from IPython.display import Image
import pydotplus
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state= 42)
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
dt = DecisionTreeClassifier()
dt.fit(X_train,y_train)
print('Decision Tree Classifer Created')
dt.predict(X_test)
tree.plot_tree(dt)
dt_score_train = dt.score(X_train,y_train)
print('Training Data score:',dt_score_train)
dt_score_test = dt.score(X_test,y_test)
print('Testing Data score:',dt_score_test)
from sklearn import tree
import matplotlib.pyplot as plt
X.columns
fn = list(X.columns)
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (7,7), dpi=1000)
tree.plot_tree(dt,feature_names = fn,filled = True)
fig.savefig('irisDT.png')
