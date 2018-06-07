# William Daniel Taylor
# 6.7.18
# using the iris dataset to visualize decision trees

# import dataset
from sklearn.datasets import load_iris
iris = load_iris()

# import numpy to remove training data later on
import numpy as np

# import decision tree classifier
from sklearn import tree

#feature names
#print iris.feature_names

#flower names
#print iris.target_names

#flower name and measurements for the first flower
#print iris.target[0]	# label of 0 indicates a setosa
#print iris.data[0]		# returns [sepal length, sepal width, petal length, petal width]

# separate data to use as test data
test_idx = [0, 50, 100] # one of each flower; first setosa is at 0, first versicolor at 50, first virginica at 100

# training data
train_data = np.delete(iris.data, test_idx, axis = 0)
train_target = np.delete(iris.target, test_idx)

# testing data
test_data = iris.data[test_idx]
test_target = iris.target[test_idx]

# create and train classifier
clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_data, train_target)

# test classifier
print("Expected output: ")
print(test_target)

print("Predictions: ")
print(clf.predict(test_data))

# code to visualize the tree's processes
from sklearn.externals.six import StringIO
import pydotplus
dot_data = StringIO()
tree.export_graphviz(clf, 
                     out_file = dot_data, 
                     feature_names = iris.feature_names, 
                     class_names = iris.target_names, 
                     filled = True, 
                     rounded = True,
                     impurity = False)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("/home/wdtay/Programming/Projects/Misc/irisDecisionTree/viz.pdf")