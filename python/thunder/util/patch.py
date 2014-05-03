"""
Utilities for converting data into patches defined by keys
"""

from numpy import transpose, zeros, ceil

from thunder.util.load import getdims


def ind_to_array(inds, shape, time_first=False):
    """ Given a list of index/value tuples, where the values are
    lists of equal length, create a numpy ndarray and set the
    values at the given indices to the given values. Used for
    chopping up n-dim data in Spark into distributed patches. """
    shape = shape + (max([len(x[1]) for x in inds]),)
    arr = zeros(shape)
    ndims = len(shape)
    for i in range(len(inds)):
        arr[inds[i][0]] = inds[i][1]
    if time_first:
        arr = transpose(arr,[ndims-1]+range(ndims-1))
    return arr


def filter_index(k, v, ranges):
    """ Given a tuple k, a value v, and a list of ranges, one list for
    each element of k, recursively computes a list that gives back the index
    of all the ranges that the elements of k fall into, as well as the elements
    of k transformed from a global coordinate system to one originating at the first
    element of that range. As a concrete example, if the tuple k is (5,10,15),
    and the list of ranges is [[(0,10),(5,15),(10,20)],[(0,10),(5,15)],[(10,20),(20,30)]],
    then the returned list of range indices is [(0,1,0),(1,1,0)] and the corresponding
    transformed indices are (5,5,5) and (0,5,5) """

    assert len(k) == len(ranges)
    if len(k) is 1:
        return [((i,), ((k[0]-x[0],), v)) for i, x in enumerate(ranges[0]) if x[1] > k[0] >= x[0]]
    else:
        foo = filter_index(k[:-1], v, ranges[:-1])
    return [(y[0] + (i,), (y[1][0] + (k[-1]-x[0],), v)) for i, x in enumerate(ranges[-1])
            for y in foo if x[1] > k[-1] >= x[0]]


def patch(data, patch_sizes, border_widths=None):
    """ Given a Spark collection in the format used by Thunder,
    where the key is the index of a time series and the value
    is the time series itself, this chops the data into patches
    of a given size, so Spark can perform operations in space
    as well as in time. Also creates a border around the patches
    to avoid edge effects from chopping up the data """

    dims = getdims(data)

    if border_widths is None:
        patches = data.map(lambda (k, v): (tuple([(x-1)/y for x, y in zip(k, patch_sizes)]), [(tuple([(x-1) % y for x, y in zip(k, patch_sizes)]), v)])) \
            .reduceByKey(lambda a, b: a+b) \
            .map(lambda (k, v): (k, ind_to_array(v, patch_sizes, time_first=True)))
    else:
        ranges = [[(w*patch_size-border_width+dim_min, w*patch_size+border_width+patch_size+dim_min)
                   for w in range(int(ceil(1.0 * (dim_max-dim_min)/patch_size)))]
                  for dim_min, dim_max, patch_size, border_width in zip(dims.mim, dims.max, patch_sizes, border_widths)]
        patches = data.flatMap(lambda (k, v): filter_index(k, v, ranges)) \
            .map(lambda (k, v): (k, [v])) \
            .reduceByKey(lambda a, b: a+b) \
            .map(lambda (k, v): (k, ind_to_array(v, tuple([patch_size+2*border_width for patch_size, border_width
                                                           in zip(patch_sizes, border_widths)]), time_first=True)))

    return patches