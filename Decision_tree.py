''' book : hands on machine learning with scikit-learn, keras and tensorflow
    chapter 6 : Decision Tree
'''
import numpy as np
import pandas as pd
from sklearn.datasets  import make_moons

X, y = make_moons(n_samples=10000, noise=0.4, random_state=42)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn.tree import DecisionTreeClassifier
decision_tree = DecisionTreeClassifier(random_state=42)

from sklearn.model_selection import GridSearchCV
param_grid = {
    'criterion' : ['entropy', 'gini'],
    'max_depth' : [None, 2, 5, 10, 20, 30],
    'min_samples_split' : [2, 5, 10, 20],
    'min_samples_leaf' : [1, 5, 10]
}

'''
to find optimized hyperparamters for a model , we use GridSearchCV
grid_search = GridSearchCV(decision_tree, param_grid=param_grid, cv=2, scoring='f1',verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)
'''

decision_tree1 = DecisionTreeClassifier(criterion='entropy', max_depth=2, min_samples_leaf=1, min_samples_split= 2)
decision_tree1.fit(X_train, y_train)
y_pred = decision_tree1.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)

'''
print(f"accuracy of model is : { accuracy }")
accuracy is 0.84 means 84%
'''

'''
to visualize decision tree, use plot_tree
'''
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
plot_tree(
    decision_tree1,
    filled=True, 
    rounded=True
)
plt.show()