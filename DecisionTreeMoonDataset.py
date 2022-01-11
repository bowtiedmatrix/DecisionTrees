from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# making a dummy dataset
X, y = make_moons(n_samples=10000, noise=0.4)

# splitting into training and tests datasets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1) 

# making the model
dtc = DecisionTreeClassifier(max_leaf_nodes=8)

# using grid search with cross-validation
parameters = {'max_leaf_nodes': [8]}		# try decreasing this to reduce overfitting
scores = cross_val_score(dtc, X, y, cv=5)
search = GridSearchCV(dtc, param_grid=parameters, scoring=scores)

dtc.fit(x_train, y_train)	# training the model

yhat = dtc.predict(x_test)
accuracy = accuracy_score(y_test, yhat)
print("Accuracy: %s" % accuracy)