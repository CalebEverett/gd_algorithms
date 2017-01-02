# Define gradient generator function for Adagrad
# [Duchi, J., Hazan, E., and Singer, Y. Adaptive Subgradient Methods for Online Learning and Stochastic Optimization](http://stanford.edu/~jduchi/projects/DuchiHaSi10_colt.pdf)

def adagrad(alpha, grad_fun):

    def theta_gen_adagrad(theta):
        # Initialize gradient history and hyperparameter epsilon
        grad_hist = 0
        epsilon = 10**-8

        # Generate gradient
        while True:
            # Get gradient to adapt from gradient function
            gradient = grad_fun(theta)

            # Perform gradient adaptation
            grad_hist += np.square(gradient)
            theta = theta - alpha * gradient / (epsilon + np.sqrt(grad_hist))
            yield theta

    return theta_gen_adagrad
