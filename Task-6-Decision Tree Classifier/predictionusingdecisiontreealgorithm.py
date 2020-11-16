# Importing Libraries in Python
import sklearn
from sklearn import datasets
import sklearn.datasets as datasets
import pandas as pd
import numpy as np
import seaborn as sns
print('Libraries Imported')

# Loading the iris dataset
iris=datasets.load_iris()

# Forming the iris dataframe
X=pd.DataFrame(iris.data, columns=iris.feature_names)

#The first 5 rows:
print(X.head())

# The last 5 rows:
print(X.tail())

#Providing the datatype and the information of the dataset.
print('Dataset Info:{}'.format(X.info()))

#Summary of the dataset:
print('Dataset Description:\n\n{}'.format(X.describe()))

#Determining the unique values present in the dataset:
X.nunique()

#Determining the null values in the dataset:
X.isnull().sum()

#Showing the correlation between the different columns:
X.corr()

#Showing the correlation in a visual format:
sns.heatmap(X.corr(), annot=True)

Y=iris.target 
print(Y)

# Importing the Required Libraries
from io import StringIO
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import metrics
from IPython.display import Image
import pydotplus

# Splitting the Data into Train and Test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state= 42)

# Defining the decision tree algorithm
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
dt = DecisionTreeClassifier()
dt.fit(X_train,y_train)
print('Decision Tree Classifer Created')

dt.predict(X_test)

tree.plot_tree(dt)

#Determining the training data score
dt_score_train = dt.score(X_train,y_train)
print('Training Data score:',dt_score_train)
#Determining the test data score.
dt_score_test = dt.score(X_test,y_test)
print('Testing Data score:',dt_score_test)

from sklearn import tree
import matplotlib.pyplot as plt

X.columns

#Graphically visualizing the Decision Tree Classfier model we made.
fn = list(X.columns)
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (7,7), dpi=1000)
tree.plot_tree(dt,feature_names = fn,filled = True)
#Saving the figure as a png image.
fig.savefig('irisDT.png') 