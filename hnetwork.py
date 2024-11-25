from typing import Literal

import numpy as np
import numpy.typing as npt

class HNetwork():


    def __init__(self, n_neurons: int, 
                 train_rule: Literal['hebb', 'oja'] = 'hebb',
                 update_mode: Literal['sync', 'async'] = 'async') -> None:
        self.n_neurons = n_neurons
        self.train_rule = train_rule
        self.update_mode = update_mode


    def fit(self, X: npt.NDArray, epochs: int = 10, lr: float = 1e-3, verbose: int = 0) -> None:
        # X is assumed to be a M (rows) by N (columns) matrix where
        # each row is a pattern of lenght N
        assert(X.shape[1] == self.n_neurons)
        if self.train_rule == 'hebb':
            self._fit_hebb(X, verbose)
        elif self.train_rule == 'oja':
            self._fit_oja(X, epochs, lr, verbose)


    def _fit_hebb(self, X : npt.NDArray, verbose: int = 0) -> None:
        M = X.shape[0]
        self.weights = (X.T @ X) / M
        np.fill_diagonal(self.weights, 0)


    def _fit_oja(self, X : npt.NDArray, epochs: int = 10, lr: float = 1e-3, verbose: int = 0):
        self.weights = np.zeros(shape=(self.n_neurons, self.n_neurons))
        np.fill_diagonal(self.weights, 0)
        for epoch in range(epochs):
            if verbose > 0:
                print(f"epoch: {epoch + 1}")
            for i, pattern in enumerate(X):
                if verbose > 1:
                    print(f"pattern: {i + 1}")
                pattern = pattern.reshape(-1, 1)
                delta_w = lr * (np.dot(pattern, pattern.T) - self.weights @ pattern @ pattern.T)
                self.weights += delta_w
                np.fill_diagonal(self.weights, 0)
                if verbose > 2:
                    print(f"max |dw|: {np.max(np.abs(delta_w))}")
                if np.max(np.abs(delta_w)) < 1e-5:
                    print(f"early stopping after {epoch} epochs")
                    return


    def predict(self, x: npt.NDArray, epochs: int = 1) -> npt.NDArray:
        state = x.flatten()
        for _ in range(epochs):
            if self.update_mode == 'sync':
                state = self._predict_synchronous(state)
            elif self.update_mode == 'async':
                state = self._predict_asynchronous(state)
        return state
    
    def _predict_synchronous(self, x: npt.NDArray) -> npt.NDArray:
        y = x @ self.weights
        return np.where(y >= 0, 1, -1)
    
    def _predict_asynchronous(self, x: npt.NDArray) -> npt.NDArray:
        order = np.random.permutation(self.n_neurons)
        for i in order:
            y = x @ self.weights[:, i]
            x[i] = 1 if y >= 0 else -1
        return x


def main():
    X = np.genfromtxt('./data/large-25x25.csv', delimiter=',')
    M, N = X.shape
    shape=(25, 25)
    hn = HNetwork(N, train_rule='oja', update_mode='synchronous')
    hn.fit(X, lr=1e-3, epochs=10)


if __name__ == '__main__':
    main()