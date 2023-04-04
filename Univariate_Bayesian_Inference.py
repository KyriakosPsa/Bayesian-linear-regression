import numpy as np
import matplotlib.pyplot as plt


class Univariate_Bayesian_Inference():
    """Univariate_Bayesian_Inference contains methods for generating samples from a univariate normal distribution,
        estimating the mean and variance of the posterior distribution 
        and plotting the prior, generating, and posterior distributions."""

    def __init__(self, prior_mean, prior_var, generating_mean, generating_var):
        self.prior_mean = prior_mean
        self.prior_var = prior_var
        self.generating_mean = generating_mean
        self.generating_var = generating_var

    # Since we dont want to caputre specifically evenly separated points as the points in self.generative_x below ,
    # we utilizie the numpy random.normal function
    def generating_distribution_sampling(self, draws):
        # draws is the number of samples we want to generate
        generated_samples = np.random.normal(
            self.generating_mean, (self.generating_var)**(1/2), size=draws)
        return generated_samples

    # We get points to plot the pdf, using evenly seperated points for the x axis that fit the range of the distribution and
    # their corresponding density based on the Gaussian function
    def generate_distribution_points(self, mean, variance):
        sigma = (variance)**(1/2)
        # We use 100 observation/target pairs to plot the pdf
        x_points = np.linspace(mean - 3*sigma, mean + 3*sigma, 1000)
        y_points = 1/(np.sqrt(2*np.pi)*sigma) * \
            np.exp(-(x_points - mean)**2/(2*variance))
        return x_points, y_points

    # We get the points to plot the prior distribution
    def prior_distribution_points(self):
        self.prior_x, self.prior_y = self.generate_distribution_points(
            self.prior_mean, self.prior_var)
        return self.prior_x, self.prior_y

    # We get the points to plot the generating distribution
    def generative_distribution_points(self):
        self.generative_x, self.generative_y = self.generate_distribution_points(
            self.generating_mean, self.generating_var)
        return self.generative_x, self.generative_y

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

    # We get the points to plot the posterior distribution
    def posterior_distribution_points(self):
        self.posterior_x, self.posterior_y = self.generate_distribution_points(
            self.posterior_mean, self.posterior_var)
        return self.posterior_x, self.posterior_y

    def plot_pdfs(self, generated_samples):
        plt.figure(figsize=(9, 6))
        plt.title(
            f"Maximum a posteriori estimation of posterior probability of the mean given N={generated_samples.shape[0]} data-points", fontsize=14)
        plt.scatter(generated_samples, np.zeros(generated_samples.shape[0]),
                    color="tab:red", label="Sample data", alpha=0.5)
        # We plot the distributions based on the points we have calculated
        plt.plot(self.generative_x, self.generative_y,
                 color="tab:blue", label="Generating Distribution", linewidth=3)
        plt.plot(self.prior_x, self.prior_y,
                 color="tab:green", label="Prior Distribution of the mean", linewidth=3)
        plt.plot(self.posterior_x, self.posterior_y,
                 color="tab:orange", label="Posterior Distribution of the estimated mean", linewidth=3)
        # Vertical lines to the denote the means of each distribution
        plt.vlines(7, 0, 4, linestyles='dashed', label="True mean",
                   color="tab:blue", linewidth=1, alpha=0.8)
        plt.vlines(0, 0, 4, linestyles='dashed', label="Prior mean",
                   color="tab:green", linewidth=1, alpha=0.8)
        plt.vlines(self.posterior_mean, 0, 4, linestyles='dashed', color="tab:orange",
                   label="Estimated posterior mean", linewidth=1, alpha=0.8)
        # Misc settings
        plt.legend(fontsize=10)
        plt.grid(alpha=0.3)
        plt.ylim(0, 1.75)
        plt.ylabel("Density", fontsize=14)
        plt.xlabel("x", fontsize=14)
        plt.xticks(np.arange(-8, 22, 2))
        plt.yticks(np.arange(0, 1.75, 0.1))
        plt.show()
