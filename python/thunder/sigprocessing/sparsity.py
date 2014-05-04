from numpy import zeros, maximum, transpose, reshape, prod, sum, nan_to_num, sign, dot
from numpy.linalg import norm, eig
from scipy.ndimage.filters import gaussian_filter
from math import sqrt

def ista(data, prox, Omega, A, lam, L, x0=None, tol=1e-6, iters=1000, verbose=False):
	""" Iterative Soft Threshold Algorithm for solving min_x 1/2*||Ax-B||_F^2 + lam*Omega(x)
		Input:
			data - matrix of the data B in the regularized optimization
			prox - proximal operator of the regularizer Omega
			Omega - regularizer
			A    - linear operator applied to x. The named argument 'transpose' determines
				   whether to apply the argument or its transpose. If a list of booleans is
				   given, the corresponding operators are applied in order (if possible)
			lam  - regularization parameter
			L    - Lipschitz constant. Should be the largest eigenvalue of A^T*A.
		  Optional:
			x0   - Initialization of x
			tol  - tolerance
			iters - maximum number of iterations
			verbose - print progress if true 
		Output:
			xk - the final value of the iterative minimization """	
	if x0 is None:
		x0 = zeros(A(data,transpose=True).shape)
	sz = x0.shape
	xk  = x0
	v = 2/L*A(data,transpose=True)
	for i in range(iters):
		xk  = prox(xk - 2/L*A(xk,transpose=[False,True]) + v, lam/L)
		if verbose:
			resid = A(xk,transpose=False)-data
			if len(sz) < 2:
				resid = reshape(resid,sz[0],prod(sz[1:])) 
			loss = norm(resid,ord='fro')**2
			reg  = Omega(xk)
			print '%d: Obj = %g, Loss = %g, Reg = %g, Norm = %g' % (i, 05*loss + lam*reg, loss, reg, norm(xk))
	return xk

def fista(data, prox, Omega, A, lam, L, x0=None, tol=1e-6, iters=1000, verbose=False):
	""" Fast Iterative Soft Threshold Algorithm for solving min_x 1/2*||Ax-B||_F^2 + lam*Omega(x)
		Input:
			data - matrix of the data B in the regularized optimization
			prox - proximal operator of the regularizer Omega
			Omega - regularizer
			A    - linear operator applied to x. The named argument 'transpose' determines
				   whether to apply the argument or its transpose. If a list of booleans is
				   given, the corresponding operators are applied in order (if possible)
			lam  - regularization parameter
			L    - Lipschitz constant. Should be the largest eigenvalue of A^T*A.
		  Optional:
			x0   - Initialization of x
			tol  - tolerance
			iters - maximum number of iterations
			verbose - print progress if true 
		Output:
			xk - the final value of the iterative minimization """
	tk1 = 1
	if x0 is None:
		x0 = zeros(A(data,transpose=True).shape)
	sz = x0.shape
	yk1 = x0
	xk  = x0
	v = 2/L*A(data,transpose=True)
	for i in range(iters):
		yk  = yk1
		xk1 = xk
		tk  = tk1

		xk  = prox(yk - 2/L*A(yk,transpose=[False,True]) + v, lam/L)
		tk1 = (1+sqrt(1+4*(tk**2)))/2
		yk1 = xk + (tk-1)/tk1*(xk-xk1)
		if verbose:
			resid = A(xk,transpose=False)-data
			if len(sz) > 2:
				resid = reshape(resid,(sz[0],prod(sz[1:])))
			loss = norm(resid,ord='fro')**2
			reg  = Omega(xk)
			print '%d: Obj = %g, Loss = %g, Reg = %g, Norm = %g' % (i, 05*loss + lam*reg, loss, reg, norm(xk))
	return xk

def gaussian_group_lasso(data, sig, lam, tol=1e-6, iters=1000, verbose=False):
	def A(data,transpose=False):
		if type(transpose) is bool:
			# Conveniently, the transpose of a gaussian filter matrix is a gaussian filter matrix
			return gaussian_filter(data,(0,) + sig,mode='wrap')
		elif type(transpose) is list:
			return gaussian_filter(data,tuple([sqrt(len(transpose))*x for x in (0,) + sig]),mode='wrap')
		else:
			raise NameError('Transpose must be bool or list of bools')
	return fista(data, \
		lambda x,t: nan_to_num(maximum(1-t/norm(x,ord=2,axis=0),0)*x), \
		lambda x:   sum(norm(x,ord=2,axis=0)), \
		A, lam, 2.0, tol=tol, iters=iters, verbose=verbose)

def gaussian_lasso(data, sig, lam, tol=1e-6, iters=1000, verbose=False):
	def A(data,transpose=False):
		if type(transpose) is bool:
			# Conveniently, the transpose of a gaussian filter matrix is a gaussian filter matrix
			return gaussian_filter(data,(0,)+sig,mode='wrap')
		elif type(transpose) is list:
			return gaussian_filter(data,tuple([sqrt(len(transpose))*x for x in (0,) + sig]),mode='wrap')
		else:
			raise NameError('Transpose must be bool or list of bools')
	return fista(data, \
		lambda x,t: maximum(abs(x)-t,0)*sign(x), \
		lambda x:   sum(abs(x)), \
		A, lam, 2.0, tol=tol, iters=iters, verbose=verbose)

def A_factory(A):
	def A_op(data, transpose=False):
		if type(transpose) is bool:
			if transpose:
				return dot(A.transpose(),data)
			else:
				return dot(A,data)
		elif type(transpose) is list:
			if len(transpose) is 1:
				return A_op(data,transpose[0])
			else:
				return A_op(A_op(data,transpose[0]),transpose[1:])
	return A_op

def lasso(data, A, lam, tol=1e-6, iters=1000, verbose=False):
	if A.shape[0] > A.shape[1]:
		L = 2*max(eig(dot(A.transpose(),A))[0])
	else:
		L = 2*max(eig(dot(A,A.transpose()))[0])
	return fista(data, \
		lambda x,t: maximum(abs(x)-t,0)*sign(x), \
		lambda x:   sum(abs(x)), \
		A_factory(A), lam, L, tol=tol, iters=iters, verbose=verbose)

def group_lasso(data, A, lam, tol=1e-6, iters=1000, verbose=False):
	""" Group lasso for the case where each group is a row of the solution matrix """
	if A.shape[0] > A.shape[1]:
		L = 2*max(eig(dot(A.transpose(),A))[0])
	else:
		L = 2*max(eig(dot(A,A.transpose()))[0])
	return fista(data, \
		lambda x,t: nan_to_num(maximum(1-t/norm(x,ord=2,axis=1),0)*x.transpose()).transpose(), \
		lambda x:   sum(norm(x,ord=2,axis=1)), \
		A_factory(A), lam, L, tol=tol, iters=iters, verbose=verbose)

if __name__ == "__main__":
	from numpy.random import rand, randn, randint
	from numpy import eye

	# generate 2D model data
	T = 200
	sz = (100,100)
	sig = (5,5)
	foo = 0.1*randn(*((T,) + sz))
	bar = zeros((T,) + sz)
	for i in range(20):
		ind = tuple([randint(x) for x in sz])
		for j in range(T):
			bar[(j,)+ind] = randn()
	foo = foo + 10*gaussian_filter(bar,(0,)+sig)
	gaussian_group_lasso(foo,sig,.1,verbose=True)

	# N = 1000
	# n = 100
	# k = 10
	# d = n/N/2
	# A = randn(n,N)
	# a = randn(N,k)
	# for i in range(n):
	# 	if rand() > d:
	# 		a[i,:] = 0.0
	# b = dot(A,a)
	# lasso(b,A,1,verbose=True)
	# group_lasso(b,A,10,verbose=True,iters=10000)

	# # generate 1D model data
	# T = 20
	# sz = 10,
	# sig = 5,
	# foo = 0.1*randn(*((T,) + sz))
	# bar = zeros((T,)+sz)
	# for i in range(sz[0]):
	# 	if rand < 0.1:
	# 		for j in range(T):
	# 			bar[j,i] = 10*randn()
	# foo = foo + gaussian_filter(bar,(0,)+sig,mode='wrap')
	# foo = foo.transpose()

	# gaussian_group_lasso(foo,sig,1,verbose=True,iters=1000) # this fails!
	# A = eye(sz[0])
	# A = gaussian_filter(A,sig+(0,),mode='wrap')
	# group_lasso(foo,A,1,verbose=True,iters=1000)