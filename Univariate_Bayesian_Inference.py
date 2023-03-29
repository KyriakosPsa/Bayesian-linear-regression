import numpy as np
import matplotlib.pyplot as plt


class Univariate_Bayesian_Inference():
    def __init__(self, prior_mean, prior_var, generating_mean, generating_var):
        self.prior_mean = prior_mean
        self.prior_var = prior_var
        self.generating_mean = generating_mean
        self.generating_var = generating_var

    # Since we dont want to caputre specifically evenly separated points as the points in self.generative_x below ,
    # we utilizie the numpy random.normal function
    def generating_distribution_sampling(self, draws):
        generated_samples = np.random.normal(
            self.generating_mean, (self.generating_var)**(1/2), size=draws)
        return generated_samples

    # We get points to plot the pdf, using evenly seperated points for the x axis that fit the range of the distribution and
    # their corresponding density based on the Gaussian function
    def generative_distribution_points(self):
        generative_sigma = (self.generating_var)**(1/2)
        self.generative_x = np.linspace(
            self.generating_mean - 3*generative_sigma, self.generating_mean + 3*generative_sigma, 100)
        self.generative_y = 1/(np.sqrt(2*np.pi)*generative_sigma)*np.exp(-(
            self.generative_x - self.generating_mean)**2/(2*generative_sigma**2))
        return self.generative_x, self.generative_y

    # As above
    def prior_distribution_points(self):
        prior_sigma = (self.prior_var)**(1/2)
        self.prior_x = np.linspace(
            self.prior_mean - 3*prior_sigma, self.prior_mean + 3*prior_sigma, 100)
        self.prior_y = 1/(np.sqrt(2*np.pi)*prior_sigma) * \
            np.exp(-(self.prior_x - self.prior_mean)**2/(2*prior_sigma**2))
        return self.prior_x, self.prior_y

    # As above
    def posterior_distribution_points(self):
        try:
            posterior_sigma = (self.posterior_var)**(1/2)
            self.posterior_x = np.linspace(
                self.posterior_mean - 3*posterior_sigma, self.posterior_mean + 3*posterior_sigma, 100)
            self.posterior_y = 1/(np.sqrt(2*np.pi)*posterior_sigma)*np.exp(-(
                self.posterior_x - self.posterior_mean)**2/(2*posterior_sigma**2))
        except NameError:
            print("You cant sample from the posterior without first estimating its paremeters, use the MAP_estimator method first")
        return self.posterior_x, self.posterior_y

    def MAP_estimation(self, generated_samples):
        # Posterior mean estimation using equation (1)
        N = generated_samples.shape[0]
        x_bar = np.mean(generated_samples)
        self.posterior_mean = (N*self.prior_var*x_bar + self.generating_var *
                               self.prior_mean)/(N*self.prior_var + self.generating_var)
        # Posterior variance estimation using equation (2)
        self.posterior_var = (self.generating_var * self.prior_var) / \
            (N*self.prior_var + self.generating_var)
        return self.posterior_mean, self.posterior_var

    def plot_pdfs(self, generated_samples):
        plt.figure(figsize=(9, 6))
        plt.title(
            f"Maximum a posterior estimation of posterior probability of the mean given N={generated_samples.shape[0]} data-points", fontsize=14)
        # We plot the distributions based on the points we have calculated
        plt.plot(self.generative_x, self.generative_y,  color="tab:blue",
                 label="Generating Distribution", linewidth=3)
        plt.plot(self.prior_x, self.prior_y,  color="tab:green",
                 label="Prior Distribution of the mean", linewidth=3)
        plt.plot(self.posterior_x, self.posterior_y, color="tab:orange",
                 label="Posterior Distribution of the estimated mean", linewidth=3)
        # Vertical lines to the denote the means of each distribution
        plt.vlines(7, 0, 4, linestyles='dashed', label="True mean",
                   color="tab:blue", linewidth=1.5, alpha=0.8)
        plt.vlines(0, 0, 4, linestyles='dashed', label="Prior mean",
                   color="tab:green", linewidth=1.5, alpha=0.8)
        plt.vlines(self.posterior_mean, 0, 4, linestyles='dashed',
                   color="tab:orange", label="Estimated mean", linewidth=1.5, alpha=0.8)
        # Misc settings
        plt.legend(fontsize=11)
        # Somehow the y-axis drifts away from 0, so we limit it to 0,4 the same as the ticks
        plt.ylim(0, 4)
        plt.ylabel("Density", fontsize=14)
        plt.xlabel("x", fontsize=14)
        plt.xticks(np.arange(-8, 22, 2))
        plt.yticks(np.arange(0, 4.0, 0.5))
        plt.grid()
        plt.show()
