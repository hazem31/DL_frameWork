import numpy as np


def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s


def identity(z):
    return z

def initialize_with_zeros(dim):

    w = np.zeros(shape=dim)
    b = np.zeros(shape=(dim[0],1))
    # assert (w.shape == dim)
    # assert (isinstance(b, float) or isinstance(b, int))
    return w, b

def logLikehood_cost_grad(m , Y, A, X):
    cost = (- 1 / m) * np.sum(Y * np.log(A) + (1 - Y) * (np.log(1 - A)))  # compute cost
    dw = (1 / m) * np.dot(X, (A - Y).T).T
    db = (1 / m) * np.sum(A - Y).T
    return cost, dw, db

def cross_entropy(m,A,Y):
    cost = (- 1 / m) * np.sum(Y * np.log(A) + (1 - Y) * (np.log(1 - A)))  # compute cost
    return cost

def optimize_sgd(model,X, Y, num_iterations, learning_rate, print_cost=False , epsilion = 0.0001):
    """
    This function optimizes w and b by running a gradient descent algorithm

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps

    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.

    Tips:
    You basically need to write down two steps and iterate through them:
        1) Calculate the cost and the gradient for the current parameters. Use propagate().
        2) Update the parameters using gradient descent rule for w and b.
    """

    costs = []

    for i in range(num_iterations):

        grads, cost = model.propagate(X, Y)

        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]

        model.w = model.w - learning_rate * dw  # need to broadcast
        model.b = model.b - learning_rate * db

        costs.append(cost)

        # Print the cost every 100 training examples
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    grads = {"dw": dw,
             "db": db}

    return grads, costs


class OneLayer:

    def __init__(self, number_of_neurons, number_of_outputs=1, act_func=identity, init_func=initialize_with_zeros, cost_func=logLikehood_cost_grad):
        self.w, self.b = init_func((number_of_outputs, number_of_neurons))
        self.number_of_neurons = number_of_neurons
        self.number_of_outputs = number_of_outputs
        self.act_func = act_func
        self.cost_func = cost_func

        if number_of_outputs == 1:
            self.classes = 2
        else:
            self.classes = number_of_outputs


    def re_init(self,init_func):
        self.w ,self.b = init_func((self.number_of_outputs,self.number_of_neurons))


    def propagate(self, X, Y , type_of_y = '0'):
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

    def predict(self,X,threshold = 0.5,z_value=False):
        '''
        Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)

        Arguments:
        w -- weights, a numpy array of size (num_px * num_px * 3, 1)
        b -- bias, a scalar
        X -- data of size (num_px * num_px * 3, number of examples)

        Returns:
        Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
        '''

        #todo should a class start with a one or zero for the first class
        #num_of_classes = [i for i in range(self.classes)]
        m = X.shape[1]
        Y_prediction = np.zeros((1, m))
        #w = w.reshape(X.shape[0], 1)

        Z = np.dot(self.w , X) + self.b

        if z_value == True:
            return Z

        A = self.act_func(Z)

        if self.classes == 2:
            for i in range(A.shape[1]):
                #todo check if should be 0 or -1
                Y_prediction[0, i] = 1 if A[0, i] > threshold else 0

        else:
            for i in range(A.shape[1]):
                #todo check this later
                Y_prediction[0, i] = np.argmax(A[:,i])

        return Y_prediction

    def train(self ,X_train , Y_train,num_iterations=2000, learning_rate=0.5, print_cost=False):
        """
        Builds the logistic regression model by calling the function you've implemented previously

        Arguments:
        X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
        Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
        X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
        Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
        num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
        learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
        print_cost -- Set to true to print the cost every 100 iterations

        Returns:
        d -- dictionary containing information about the model.
        """


        # Gradient descent (≈ 1 line of code)
        grads , costs = optimize_sgd(self, X_train, Y_train, num_iterations, learning_rate, print_cost)

        # Predict test/train set examples (≈ 2 lines of code)

        #Y_prediction_test = self.predict(X_test)
        #Y_prediction_train = self.predict(X_train)



        #print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
        #print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

        d = {"costs": costs,
             "w": self.w,
             "b": self.b,
             "learning_rate": learning_rate,
             "num_iterations": num_iterations}

        return d

    def accuracy(self,X,Y):
        prediction = self.predict(X)
        accuracy = 100 - np.mean(np.abs(prediction - Y)) * 100
        return accuracy

    def test(self,X_test,Y_test):
        Y_prediction_test = self.predict(X_test)
        accuracy = 100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100
        return accuracy

# model = OneLayer(2, act_func=sigmoid)
# X, Y = np.array([[1,2], [3,4]]), np.array([[1, 0]])
# model.w = np.array([[1], [2]]).T
# model.b = 2
# grad ,cost = model.propagate(X, Y)
#
# print(cost)
# print(grad)
#
# print(model.predict(X))
#
# grads, costs = optimize_sgd(model, X, Y, num_iterations= 100, learning_rate = 0.009, print_cost = False)
#
# print("w = " + str(model.w))
# print("b = " + str(model.b))
# print("dw = " + str(grads["dw"]))
# print("db = " + str(grads["db"]))


def tanh(z):
    return np.tanh(z)

def random_init_zero_bias(n_2,n_1 , mult = 0.01):
    return np.random.randn(n_2, n_1) * 0.01 , np.zeros(shape=(n_2, 1))


class MultiLayer:

    def __init__(self, number_of_neurons = 0, cost_func=cross_entropy):
        self.w, self.b = [] , []
        self.layer_size = []
        self.number_of_input_neurons = number_of_neurons
        self.number_of_outputs = 0
        self.act_func = []
        self.act_func_out = None
        self.cost_func = cost_func

        self.parameters = {}
        self.cache = {}
        self.prev = []

    def addLayerInput(self,size):
        self.number_of_input_neurons = size
        self.layer_size.append(size)


    def addHidenLayer(self,size,act_func=sigmoid):
        self.layer_size.append(size)
        self.act_func.append(act_func)

    def addOutputLayer(self,size,act_func=sigmoid):
        self.number_of_outputs = size
        self.layer_size.append(size)
        self.act_func.append(act_func)


    def initialize_parameters(self,seed = 2 , init_func = random_init_zero_bias):
        """
        Argument:
        n_x -- size of the input layer
        n_h -- size of the hidden layer
        n_y -- size of the output layer

        Returns:
        params -- python dictionary containing your parameters:
                        W1 -- weight matrix of shape (n_h, n_x)
                        b1 -- bias vector of shape (n_h, 1)
                        W2 -- weight matrix of shape (n_y, n_h)
                        b2 -- bias vector of shape (n_y, 1)
        """

        np.random.seed(seed)  # we set up a seed so that your output matches ours although the initialization is random.


        for i in range(len(self.layer_size)-1):
            out = init_func(self.layer_size[i+1] , self.layer_size[i])
            self.w.append(out[0])
            self.b.append(out[1])



        for i in range(len(self.layer_size)-1):
            self.parameters["W"+str(i+1)] = self.w[i]
            self.parameters["b"+str(i+1)] = self.b[i]

        return self.parameters

    def forward_propagation(self,X):
        """
        Argument:
        X -- input data of size (n_x, m)
        parameters -- python dictionary containing your parameters (output of initialization function)

        Returns:
        A2 -- The sigmoid output of the second activation
        cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
        """
        # Retrieve each parameter from the dictionary "parameters"
        ### START CODE HERE ### (≈ 4 lines of code)
        # W1 = parameters['W1']
        # b1 = parameters['b1']
        # W2 = parameters['W2']
        # b2 = parameters['b2']
        # ### END CODE HERE ###

        # Implement Forward Propagation to calculate A2 (probabilities)
        ### START CODE HERE ### (≈ 4 lines of code)

        self.prev.append((1,X))
        for i in range(len(self.layer_size) - 1):
            Zi = np.dot(self.w[i], self.prev[i][1]) + self.b[i]
            Ai = self.act_func[i](Zi)
            self.prev.append((Zi,Ai))


        A_last = self.prev[-1][1]


        for i in range(len(self.layer_size)-1):
            self.cache["Z"+str(i+1)] = self.prev[i+1][0]
            self.cache["A" + str(i + 1)] = self.prev[i + 1][1]


        #todo sould i compute cost in here

        return A_last, self.cache

    def set_cost(self,cost_func):
        self.cost_func = cost_func


    def compute_cost(self,Alast, Y):
        m = Alast.shape[1]
        return self.cost_func(m,Alast,Y)

    def backward_propagation(self, X, Y):
        """
        Implement the backward propagation using the instructions above.

        Arguments:
        parameters -- python dictionary containing our parameters
        cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".
        X -- input data of shape (2, number of examples)
        Y -- "true" labels vector of shape (1, number of examples)

        Returns:
        grads -- python dictionary containing your gradients with respect to different parameters
        """
        m = X.shape[1]

        # First, retrieve W1 and W2 from the dictionary "parameters".
        ### START CODE HERE ### (≈ 2 lines of code)
        W1 = self.w[0]
        W2 = self.w[1]
        ### END CODE HERE ###

        # Retrieve also A1 and A2 from dictionary "cache".
        ### START CODE HERE ### (≈ 2 lines of code)
        A1 = self.cache['A1']
        A2 = self.cache['A2']
        ### END CODE HERE ###

        # Backward propagation: calculate dW1, db1, dW2, db2.
        ### START CODE HERE ### (≈ 6 lines of code, corresponding to 6 equations on slide above)
        dZ2 = A2 - Y
        dW2 = (1 / m) * np.dot(dZ2, A1.T)
        db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
        dZ1 = np.multiply(np.dot(W2.T, dZ2), 1 - np.power(A1, 2))
        dW1 = (1 / m) * np.dot(dZ1, X.T)
        db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
        ### END CODE HERE ###

        #todo all depends on the typr of function
        grad_list = []
        dzi = self.prev[-1][1] - Y
        for i in range(len(self.w),0,-1):
            dwi = (1 / m) * np.dot(dzi, self.prev[i-1][1].T)
            dbi = (1 / m) * np.sum(dzi, axis=1, keepdims=True)
            dzi = np.multiply(np.dot((self.w[i-1]).T, dzi), 1 - np.power(A1, 2))



        grads = {"dW1": dW1,
                 "db1": db1,
                 "dW2": dW2,
                 "db2": db2}

        return grads
