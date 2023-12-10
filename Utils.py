from sklearn.metrics import mean_squared_error
import numpy as np

class Utils:
    def root_mean_squared_error(test_data, predictions):
        return np.sqrt(mean_squared_error(test_data, predictions))

    def quantile_loss(y_true, y_pred, alpha):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        epsilon = 1e-10
        e = y_true - y_pred
        return np.sum(np.where(e >= 0, alpha * e, (alpha - 1) * e))