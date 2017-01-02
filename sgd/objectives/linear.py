# Define gradient generator function
# batch is tuple of (X values, y values)
def grad_fun_linear(theta, batch):
    X, y = batch
    return X.T.dot(X.dot(theta) - y) / X.shape[0]

def cost_fun_linear(theta, data):
    X = data[:,:-1]
    y = data[:,-1]
    return np.append(theta, np.sum(np.square(X.dot(theta) - y)) / (2 * X.shape[0]))
