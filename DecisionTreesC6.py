# Training and visualising a decision tree

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from graphviz import Source

iris = load_iris()
X = iris.data[:, 2:]	# petal length and width
y = iris.target

tree_clf = DecisionTreeClassifier(max_depth=2)
tree_clf.fit(X, y)

with open("C:/Users/Nyambura/Documents/Coding/iris_tree.dot", "w") as f:
	export_graphviz(
		tree_clf,
		out_file=f,
		feature_names=iris.feature_names[2:],
		class_names=iris.target_names,
		rounded=True,
		filled=True
		)

# dot_path = "C:/Users/Nyambura/Documents/Coding/iris_tree.dot"
# output = Source.from_file(dot_path, format='png')
# output.view()

tree_clf.predict_proba([[5, 1.5]])	# returns an array of the probabilities for each class
tree_clf.predict([[5, 1.5]])	# returns the index of the predicted class