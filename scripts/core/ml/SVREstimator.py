from core.ml.Estimator import Estimator
from sklearn.svm import SVR


# Use GridSearchCV (https://scikit-learn.org/stable/modules/generated/
# sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV)
# to find the most optimal SVR parameters!
# example: https://scikit-learn.org/stable/auto_examples/plot_kernel_ridge_regression.html
# More to read: https://scikit-learn.org/stable/modules/grid_search.html#grid-search (!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!)
class SVREstimator(Estimator):
    def __init__(self, x_train, x_test, y_train, y_test, sat_name,
                 kernel='rbf', C=1e3, gamma=0.1, epsilon=0.1, degree=3):
        Estimator.__init__(self, x_train, x_test, y_train, y_test, sat_name)
        self.regressor = SVR(kernel=kernel, C=C, gamma=gamma, epsilon=epsilon, degree=degree)

    def calculate_fitness(self):
        self.fitness = self.regressor.score(self.x_test, self.y_test)

    # TODO: implement GridSearchCV and SimulatedAnnealing
    def optimize_parameters(self):
        pass
