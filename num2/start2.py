import numpy as np
import matplotlib.pyplot as plt
from testCases_v2 import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets
import frameWork1 as fw

np.random.seed(1)

X, Y = load_planar_dataset()
print(Y.shape)

# plt.scatter(X[0, :], X[1, :], c=Y[0,:], s=40, cmap=plt.cm.Spectral);
# plt.show()

# shape_X = X.shape
# shape_Y = Y.shape
# m = X.shape[1]
#
# print ('The shape of X is: ' + str(shape_X))
# print ('The shape of Y is: ' + str(shape_Y))
# print ('I have m = %d training examples!' % (m))


# clf = sklearn.linear_model.LogisticRegressionCV();
# clf.fit(X.T, Y.T);
#
# plot_decision_boundary(lambda x: clf.predict(x), X, Y)
# plt.title("Logistic Regression")
# plt.show()

# LR_predictions = clf.predict(X.T)
# print ('Accuracy of logistic regression: %d ' % float((np.dot(Y, LR_predictions) + np.dot(1 - Y,1 - LR_predictions)) / float(Y.size) * 100) +
#        '% ' + "(percentage of correctly labelled datapoints)")


#
# model = fw.MultiLayer()
#
# model.addLayerInput(X.shape[0])
#
# model.addHidenLayer(4)
#
# model.addOutputLayer(Y.shape[0])
#
# X_assess, Y_assess = layer_sizes_test_case()
# print("The size of the input layer is: n_x = " + str(model.layer_size[0]))
# print("The size of the hidden layer is: n_h = " + str(model.layer_size[1]))
# print("The size of the output layer is: n_y = " + str(model.layer_size[2]))
#


# n_x, n_h, n_y = initialize_parameters_test_case()
#
# model = fw.MultiLayer()
#
# model.addLayerInput(n_x)
#
# model.addHidenLayer(n_h,fw.tanh)
#
# model.addOutputLayer(n_y,fw.sigmoid)
#
# parameters = model.initialize_parameters(seed=2 , init_func=fw.random_init_zero_bias)
#
#
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))
#
# print(parameters['b1'].shape)


# model = fw.MultiLayer()
#
# X_assess, parameters = forward_propagation_test_case()
#
# model.addLayerInput(X_assess.shape[0])
#
# model.addHidenLayer(4,fw.tanh)
#
# model.addOutputLayer(1,fw.sigmoid)
#
# model.initialize_parameters()
#
# model.w[0] = parameters["W1"]
# model.w[1] = parameters["W2"]
# model.b[0] = parameters["b1"]
# model.b[1] = parameters["b2"]
#
# A2, cache = model.forward_propagation(X_assess)
#
# print(np.mean(cache['Z1']), np.mean(cache['A1']), np.mean(cache['Z2']), np.mean(cache['A2']))


A2, Y_assess, parameters = compute_cost_test_case()

model = fw.MultiLayer()

model.addLayerInput(X_assess.shape[0])

model.addHidenLayer(4,fw.tanh)

model.addOutputLayer(1,fw.sigmoid)

model.initialize_parameters()

model.w[0] = parameters["W1"]
model.w[1] = parameters["W2"]
model.b[0] = parameters["b1"]
model.b[1] = parameters["b2"]

model.set_cost(fw.cross_entropy)
cost = model.compute_cost(A2,Y_assess)

print("cost = " + str(cost))