"""
Class for Generalized Linear Dimensionality Reduction
"""
import abc

from numpy import random, dot, add, zeros, shape
from scipy.linalg import svd
from thunder.util import RowMatrix


# x qua matrix is (d x n)
# q qua matrix is (d x r)

# if there are more dimensions than data points (axes = 0)
# x will be a RowMatrix with nrows d and ncols n

# if there are more data points than dimensions (axes = 1)
# x will be a RowMatrix with nrows n and ncols d

# q is always a local (min(d,n) x r) array
class LDR(object):

    def __init__(self, r, axes=1, maxiter=10, alpha=0.1, linesearch=False, convergence=False):
        self.axes = axes
        self.r = r
        self.maxiter = maxiter
        self.alpha = alpha
        self.linesearch = linesearch
        self.convergence = convergence

    @staticmethod
    def train(algorithm, data, r, **opts):
        model = ALGORITHMS[algorithm](r, **opts)
        model.optimize(data)
        return model


class LDRAlgorithm(LDR):

    @abc.abstractmethod
    def objective(self, q, x):
        pass

    @abc.abstractmethod
    def gradient(self, q, x):
        pass

    @abc.abstractmethod
    def prepare(self, x):
        pass

    def optimize(self, x):

        # function for projecting onto the stiefel manifold
        def projectstiefel(q):
            u, s, v = svd(q, full_matrices=False)
            return dot(u, v.T)

        # convert to RowMatrix is not provided
        if type(x) is not RowMatrix:
            x = RowMatrix(x)

        # prepare for optimization through pre-computation
        self.prepare(x)

        # check dimensions before doing optimization
        if self.axes == 1:
            self.n = x.nrows
            self.d = x.ncols
        if self.axes == 0:
            self.n = x.ncols
            self.d = x.nrows

        # initialize a random local array
        q = projectstiefel(random.randn(min(self.d, self.n), self.r))

        # initialize array for history
        history = zeros(self.maxiter)

        # main gradient loop
        for i in range(0, self.maxiter):

            grad = self.gradient(q, x)
            z = -1 * (grad - dot(q, dot(grad.T, q)))
            q += self.alpha * z
            q = projectstiefel(q)

            if self.convergence is True:
                history[i] = self.objective(q, x)

        self.qopt = q
        self.history = history

    def transform(self, x):
        return x.times(self.qopt)


class GeneralizedPCA(LDRAlgorithm):

    def __init__(self, r, **opts):
        LDR.__init__(self, r, **opts)

    def prepare(self, x):
        self.xx = x.gramian()

    def objective(self, q, x):

        if self.axes == 1:
            qx = x.times(dot(q, q.T))
            tr = x.rdd.values().zip(qx.rdd.values()).map(lambda (x, y): x - y).map(lambda x: dot(x, x)).reduce(add)
            return tr

    def gradient(self, q, x):

        if self.axes == 1:
            grad = -2 * dot(self.xx, q)
            grad += dot(q, dot(q.T, dot(self.xx, q))) + dot(self.xx, dot(q, dot(q.T, q)))
            return grad


ALGORITHMS = {
    'pca': GeneralizedPCA
}