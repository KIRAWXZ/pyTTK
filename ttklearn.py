# quick wrapper around the ttk algorithm to give an sklearn-style interface

import ttk
import numpy as np

class TTKClassifier(Object):
    def __init__(self, k, C, max_iter=5000):
        self._w = None
        self._b = None

        self.k = k
        self.C = C
        self.MAX_ITER = max_iter

    def fit(self, x_train, y_train, x_test, init_w=None, init_b=0):
        self._w, self._b = ttk.run_ttk(x_train, y_train, x_test, self.k, self.C, 
                                        init_w=init_w, init_b=init_b, MAX_ITER=self.max_iter)

    def predict(self, x_test):
        if self._w is None:
            raise ValueError("fit() must be run before predicting")
        return [1 if (np.dot(self._w, x_i) + self._b) > 1e-6 else -1 for x_i in x_test]

    def decision_function(self, x_test):
        if self._w is None:
            raise ValueError("fit() must be run before predicting")
        return [np.dot(self._w, x_i) + self._b for x_i in x_test]

    def test_precision(self, x_test, y_test):
        if self._w is None:
            raise ValueError("fit() must be run before predicting")
        return ttk.test_precision(self._w, self._b, x_test, y_test)



