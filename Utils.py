from sklearn.metrics import mean_squared_error
import numpy as np

class Utils:
    def root_mean_squared_error(test_data, predictions):
        return np.sqrt(mean_squared_error(test_data, predictions))