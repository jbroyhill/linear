# Importing our libraries
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn import neural_network #for ANNs
from sklearn import linear_model # for linear regression
from sklearn.model_selection import train_test_split # This will split the data between train and test
from sklearn.metrics import accuracy_score # This gives us the percent of accuracy for our predictions
from sklearn.neighbors import KNeighborsClassifier # for knearstneighbor

# Turning off warnings
import warnings
warnings.filterwarnings('ignore')

# Getting .csv file
csvDataFile = r"data2.csv"
# Putting file into data frame
df = pd.read_csv(csvDataFile)
# Selecting our features
features = df[["Tax rate", "Persistent EPS", "Non-industry revenue", "R&D Expense Rate", "Total OP Margin", "Net Interest Rate", "Cash Profit Margin", "10 Year Cash Growth Rate", ]]

# Selecting our label
labels = df[['Open/Closed']]

# Splitting up our testing and training data 50/50
features_training_data, features_testing_data, labels_training_data, labels_testing_data = train_test_split(features, labels, test_size=.5)


# Training our app
classifier_decision_tree = tree.DecisionTreeClassifier()
classifier_neural_network = neural_network.MLPClassifier()
classifier_k_nearest_neighbors = KNeighborsClassifier()
linear_regression = linear_model.LinearRegression()

# Making our classifiers
classifier_decision_tree = classifier_decision_tree.fit(features_training_data, labels_training_data)
classifier_neural_network = classifier_neural_network.fit(features_training_data, np.ravel(labels_training_data))
classifier_k_nearest_neighbors = classifier_k_nearest_neighbors.fit(features_training_data, np.ravel(labels_training_data))

# Setting up linear regressions
linear_regression = linear_regression.fit(features_training_data, labels_training_data)

# Making predictions
predictions_for_classifier_decision_tree = classifier_decision_tree.predict(features_testing_data)
predictions_for_classifier_neural_network = classifier_neural_network.predict(features_testing_data)
predictions_for_classifier_k_nearest_neighbors = classifier_k_nearest_neighbors.predict(features_testing_data)
predictions_for_linear_regression = linear_regression.predict(features_testing_data)

# Printing out prediction % and info for user for our decision tree
predection_percent_score_DT = accuracy_score(labels_testing_data, predictions_for_classifier_decision_tree)
percentage = "{:.0%}​​".format(predection_percent_score_DT)
print("Decision Tree Classifier prediction score is ", percentage)

# Print out prediction % for neutral network
predection_percent_score_ANN = accuracy_score(labels_testing_data, predictions_for_classifier_neural_network)
percentage = "{:.0%}​​".format(predection_percent_score_ANN)
print("Neural network Classifier prediction score is ", percentage)

# Print out prediction % for nearest neighbors
predection_percent_score_KNN = accuracy_score(labels_testing_data, predictions_for_classifier_k_nearest_neighbors)
percentage = "{:.0%}​​".format(predection_percent_score_KNN)
print("K Nearest Neighbors prediction score is ", percentage)

# Print out prediction % for Linear Regression
# fix because reg is not a good classifier
predection_percent_score_LR = accuracy_score(labels_testing_data, predictions_for_linear_regression.round())
percentage = "{:.0%}​​".format(predection_percent_score_LR)
print("Linear Regression prediction score is ", percentage)
print("\n\n\t *** END OF PROGRAM ***\n\n\n")
