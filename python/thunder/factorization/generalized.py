
import abc

from numpy import random, dot, add, outer
from thunder.util import RowMatrix


# data qua matrix is (d x n)
# q qua matrix is (d x r)

# if there are more dimensions than data points (axes = 0)
# data will be a RowMatrix with nrows d and ncols n
# and q will be a RowMatrix with nrows d and ncols r

# if there are more data points than dimensions (axes = 1)
# data will be a RowMatrix with nrows n and ncols d
# and q will be a local array with shape (d, r)
class GeneralizedDimensionalityReduction(object):

    def __init__(self, x, r, axes=1, maxiter=10, alpha=0.1):
        self.axes = axes
        if self.axes == 1:
            self.n = x.nrows
            self.d = x.ncols
        if self.axes == 0:
            self.n = x.ncols
            self.d = x.nrows
        self.r = r
        self.maxiter = maxiter
        self.alpha = alpha
        self.x = x

    @staticmethod
    def train(algorithm, data):
        model = ALGORITHMS[algorithm](data)
        model.fit(data)
        return model


class GeneralizedDimensionalityReductionAlgorithm(GeneralizedDimensionalityReduction):

    @abc.abstractmethod
    def objective(self, q):
        pass

    @abc.abstractmethod
    def gradient(self, q):
        pass

    def fit(self, data):

        # create a random local array or RowMatrix
        if self.axes == 0:
            q = RowMatrix(data.mapValues(lambda x: random.randn(self.r)), data.nrows, self.r)
        if self.axes == 1:
            q = random.randn(100, 10)

        # main gradient loop
        for i in range(0, self.maxiter):

            g = self.gradient(data, q)

            # if axes == 1, q and g are small (both fit on driver)
            # so applying the update is trivial
            # could also probably add the line search
            # because with precomputation, recomputing the objective is fast
            if self.axes == 1:
                q += self.alpha * (dot(q, dot(g.T, q)) - g)

            # if axes == 0, q and g are rdds
            # so we compute the descent direction and apply the update with a zip
            if self.axes == 0:
                gq = g.times(q)
                q = q.zip(g).mapValues(lambda qq, gg: self.alpha * (qq + dot(qq, gq) - gg))


class GeneralizedPCA(GeneralizedDimensionalityReductionAlgorithm):

    def __init__(self, data, **opts):
        GeneralizedDimensionalityReduction.__init__(self, data, **opts)
        if self.axes == 0:
            self.xx = data.gramian()

    def objective(self, q):

        if self.axes == 1:
            qx = self.x.times(dot(q, q.T))
            return self.x.zip(qx).map(lambda (x, y): x - y).map(lambda x: dot(x, x)).reduce(add)

    def gradient(self, q):

        # once we have X * X^T this is a trivial local operation
        if self.axes == 1:
            grad = -2 * dot(self.xx, q)
            grad += dot(q, dot(q.T, dot(self.xx, q))) + dot(self.xx, dot(q, dot(q.T, q)))
            return grad

        # this one is more involved...
        if self.axes == 0:
            qx = q.times(self.x)
            qq = q.gramian()
            qxxq = dot(qx, qx)
            term1 = self.x.times(-2 * qx + dot(qx, qq))
            term2 = q.times(qxxq)
            grad = term1.elementwiseplus(term2)
            return grad


ALGORITHMS = {
    'pca': GeneralizedPCA
}