import numpy as np
import matplotlib.pyplot as plt


class Least_squares_regression():
    """Least_squares_regressionp provides functionality to create and fit a polynomial regression model using the least squares method.
      It allows users to generate sample data, construct polynomial models, fit them and predict values for new data."""

    def generate_model_points(self, N):
        # 100 points to graph a period of the true model
        x_generating = np.linspace(0, 1, 100)
        y_generating = np.sin(2*np.pi*x_generating)
        # generating model with sample of size N predictions + noise
        X_train = np.linspace(0, 1, N)
        y_train = np.sin(2*np.pi*X_train) + np.random.normal(0, 1, N)
        return X_train, y_train, x_generating, y_generating

    def polynomial_model_constructor(self, X, degree):
        Phi = []
        for i in range(X.shape[0]):
            # Create the structure of the model, based on the polynomial degree using a list comprehension
            Phi.append([1] + [X[i]**j for j in range(1, degree+1)])
        return np.array(Phi)

    # Fit the model using the least squares method to find the optimal weights
    def fit(self, Phi, y_train):
        self.w = np.dot(
            (np.dot(np.linalg.inv(np.dot(Phi.T, Phi)), Phi.T)), y_train)
        return self.w

    # Predict the values of the model for new data using the trained weights
    def predict(self, Phi):
        return np.dot(Phi, self.w)

    # Calculate the root mean squared error of the model
    def evaluate(self, y_pred, y_train):
        return np.sum((((y_pred - y_train)**2)/y_train.shape[0])**(1/2))


def fit_and_evaluate_polynomials(X_train, y_train, regression_model):
    """fit_and_evaluate_polynomials function can be used to fit
    and evaluate polynomial models of varying degrees."""
    # Build the Phi matrix per polynomial degree
    per_polynomial_Phi = []
    for degree in [2, 3, 4, 5, 9]:
        Phi = regression_model.polynomial_model_constructor(X_train, degree)
        per_polynomial_Phi.append(Phi)

    # Collect as we iterate per polynomial relevant information
    per_polynomial_w = []
    per_polynomial_predictions = []
    per_polynomial_RMSE = []
    for Phi in per_polynomial_Phi:
        # Fit
        w = regression_model.fit(Phi, y_train)
        per_polynomial_w.append([round(coef, 4) for coef in w])
        # Predict
        y_pred = regression_model.predict(Phi)
        per_polynomial_predictions.append(y_pred)
        # Evaluate
        RMSE = regression_model.evaluate(y_pred, y_train)
        per_polynomial_RMSE.append(RMSE)
    return per_polynomial_w, per_polynomial_predictions, per_polynomial_RMSE

# Plots all the fitted polynomial models created with fit_and_evaluate_polynomials, as well as the true model


def plot_fitted_models(X_train, y_train, per_polynomial_predictions, x, y):
    """plot_fitted_models function plots the curves of the fitted polynomial models and the true model."""
    plt.figure(figsize=(10, 10))
    plt.title(
        f"Least squares fitted polynomial models of varying degrees vs the true model curve for a sample dataset of N = {X_train.shape[0]}", fontsize=14)
    plt.scatter(X_train, y_train, label="Training set",
                color="tab:pink", alpha=0.8)
    for predictions, degree in zip(per_polynomial_predictions, [2, 3, 4, 5, 9]):
        plt.plot(X_train, predictions,
                 label=f"Polynomial Model of degree: {degree}", linewidth=2)
    plt.plot(x, y, label="True Model sin(2πx)",
             color="Black", linewidth=2, linestyle='-.')
    plt.ylabel("Targets", fontsize=15)
    plt.xlabel('Observations', fontsize=15)
    plt.xlim(0, 1)
    plt.legend()
    plt.grid()
    plt.show()
