# Define theta generator function for Adam
def adam(alpha, grad_fun):

    # Create theta generator
    def theta_gen_adam(theta):
        # Initialize moment variables and hyperparameters
        moment1 = np.zeros(theta.shape[0])
        moment2 = np.zeros(theta.shape[0])
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 10**-8
        t = 1

        # Generate new theta
        while True:
            # Get gradient to adapt
            gradient = grad_fun(theta)

            # Update moment estimates
            moment1 = beta1 * moment1 + (1 - beta1) * gradient
            moment2 = beta2 * moment2 + (1 - beta2) * np.square(gradient)
            moment1_hat = moment1 / (1 - beta1**t)
            moment2_hat = moment2 / (1 - beta2**t)

            # Yield adapted gradient
            theta_new = theta - alpha * moment1_hat / (epsilon + np.sqrt(moment2_hat))
            yield theta_new
            t += 1
            theta = theta_new

    return theta_gen_adam
