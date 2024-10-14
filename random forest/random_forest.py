
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

X, y = make_moons(10000, noise=0.4, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

forest_clf = RandomForestClassifier(100, max_leaf_nodes=16, n_jobs=-1)

forest_clf.fit(X_train, y_train)
