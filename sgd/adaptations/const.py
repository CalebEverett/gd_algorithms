# Define generator function for constant alpha
# Not terribly useful here, but allows other gradient adaptations
def const_alpha(alpha, grad_fun):

    def theta_gen_const(theta):
        while True:
            theta = theta - alpha * grad_fun(theta)
            yield theta

    return theta_gen_const
