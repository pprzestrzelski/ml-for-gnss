from src.core.ml.Estimator import Estimator
from sklearn.linear_model import \
    (LinearRegression, TheilSenRegressor, RANSACRegressor, HuberRegressor, SGDRegressor, Ridge)
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


# https://scikit-learn.org/stable/auto_examples/linear_model/
# plot_robust_fit.html#sphx-glr-auto-examples-linear-model-plot-robust-fit-py
class LinearEstimator(Estimator):
    """
    Linear estimators dedicated to GNSS satellite data prediction.
    Therefore some descriptions may be domain specific, e.g. plot title.
    """
    def __init__(self, x_train, x_test, y_train, y_test, sat_name, estimator="OLS", degree=1):
        Estimator.__init__(self, x_train, x_test, y_train, y_test, sat_name)
        self.estimator_name = None
        self.degree = None
        self.estimators = \
            {'OLS': LinearRegression(),
             'Theil-Sen': TheilSenRegressor(random_state=42),
             'RANSAC': RANSACRegressor(random_state=42),
             'HuberRegressor': HuberRegressor(),
             'SGD': SGDRegressor(random_state=42),
             'Ridge': Ridge(random_state=42)}
        self.set_estimator(estimator, degree)
        self.data_trained = False

    def set_estimator(self, estimator, degree=1):
        if estimator not in self.estimators:
            print("ERROR: {} is not available (tip: use available_estimators())".format(estimator))
        else:
            self.estimator_name = estimator
            self.degree = degree
            self.regressor = make_pipeline(PolynomialFeatures(self.degree), self.estimators[estimator])
            self.__clear_all()

    def fit(self):
        if not self.estimator_name:
            print("ERROR: object was not initialized correctly!")
        super().fit()
        self.data_trained = True

    def predict(self):
        if not self.data_trained:
            print("ERROR: could not make prediction, train data first!")
        super().predict()

    def available_estimators(self):
        out = []
        for name in self.estimators.keys():
            out.append(name)
        return out

    def calculate_fitness(self):
        self.fitness = self.regressor.score(self.x_test, self.y_test)

    def __clear_all(self):
        self.data_trained = False
        self.y_pred = None
        self.df = None
        self.mae = None
        self.mse = None
        self.rms = None
