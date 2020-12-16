import numpy as np


def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s


def identity(z):
    return z

def initialize_with_zeros(dim):

    w = np.zeros(shape=dim)
    b = np.zeros(shape=dim[0])
    # assert (w.shape == dim)
    # assert (isinstance(b, float) or isinstance(b, int))
    return w, b

def logLikehood_cost_grad(m, Y, A, X):
    cost = (- 1 / m) * np.sum(Y * np.log(A) + (1 - Y) * (np.log(1 - A)))  # compute cost
    dw = (1 / m) * np.dot(X, (A - Y).T).T
    db = (1 / m) * np.sum(A - Y).T
    return cost, dw, db



class OneLayer:

    def __init__(self, number_of_neurons, number_of_outputs=1, act_func=identity, init_func=initialize_with_zeros, cost_func=logLikehood_cost_grad):
        self.w, self.b = init_func((number_of_outputs, number_of_neurons))
        self.number_of_neurons = number_of_neurons
        self.number_of_outputs = number_of_outputs
        self.act_func = act_func
        self.cost_func = cost_func

    def propagate(self, X, Y):
        """
        Implement the cost function and its gradient for the propagation explained above

        Arguments:
        w -- weights, a numpy array of size (num_px * num_px * 3, 1)
        b -- bias, a scalar
        X -- data of size (num_px * num_px * 3, number of examples)
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

        Return:
        cost -- negative log-likelihood cost for logistic regression
        dw -- gradient of the loss with respect to w, thus same shape as w
        db -- gradient of the loss with respect to b, thus same shape as b

        Tips:
        - Write your code step by step for the propagation
        """
        m = X.shape[1]

        Z = np.dot(self.w, X) + self.b
        A =self.act_func(Z)

        cost, dw, db = self.cost_func(m, Y, A,X)

        grads = {"dw": dw,
                 "db": db}

        return grads, cost


model = OneLayer(2, act_func=sigmoid)
X, Y = np.array([[1,2], [3,4]]), np.array([[1, 0]])
model.w = np.array([[1], [2]]).T
model.b = 2
grad ,cost = model.propagate(X, Y)

print(cost)
print(grad)