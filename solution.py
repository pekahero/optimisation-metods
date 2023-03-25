#!/usr/bin/python

import getopt
import numpy as np
import sys
import json
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from numpy.linalg import norm
from scipy.constants import golden_ratio
from scipy.signal import savgol_filter
from scipy.linalg import ldl, lstsq, inv
from scipy.optimize import brent
from scipy.optimize.linesearch import scalar_search_wolfe2
from scipy.special import expit as sigmoid
from scipy.sparse import hstack, issparse, csc_matrix
from sklearn.datasets import load_svmlight_file
from time import perf_counter
from scipy.optimize import line_search as s_ls


class PoisRegOracle:
    def __init__(self, X, y, eps=1e-15, l1=0, l2=0, batch_size=None):
        bias = np.ones(shape=(X.shape[0], 1))
        if issparse(X):
            self.X_b = hstack([X, bias]).tocsr()
        else:
            self.X_b = np.append(X, bias, axis=1)
        
        self.y = np.int64(y)      
        self.N = y.shape[0]
        self.n_samples, self.n_features = self.X_b.shape
        self.eps = eps
        self.l1 = l1
        self.l2 = l2
        self.batch_size = batch_size
        
        
    def func(self, w):
        p = self.X_b @ w
        f = -np.mean(self.y * p - np.exp(p))
        
        return f
    
    
    def loss(self, w):
        l = self.func(w)
        l += self.l1 * norm(w, 1) + self.l2 * norm(w, 2) ** 2
        
        return l
        
    
    def grad(self, w):        
        X = self.X_b 
        y = self.y
        N = self.N
        if self.batch_size:
            X = self.X_batch
            y = self.y_batch
            N = self.batch_size

        g = -(y - np.exp(X @ w)) @ X / N
        
        return g + np.sign(w) * self.l1 + self.l2 * w
    
    
    def hess(self, w):
        W = np.diag(np.exp(self.X_b @ w).squeeze())
        return self.X_b.T @ W @ self.X_b / self.N
    
    
    def func_directed(self, w, d, alpha):
        return self.func(w + alpha * d)

    
    def grad_directed(self, w, d, alpha):
        return self.grad(w + alpha * d) @ d
    
    
    def prox(self, w, L):
        return np.sign(w) * np.maximum(np.abs(w) - L * self.l1, 0)
    
    
    def next_batch(self):
        if not self.batch_size:
            return
        
        batch = np.random.choice(self.n_samples, self.batch_size, replace=False)
        self.X_batch = self.X_b[batch] 
        self.y_batch =  self.y[batch]
    

class LineSearcher:
    def __init__(self, method='Wolfe', **kwargs):
        self._method = method
        if self._method == 'Armijo':
            self.c1 = kwargs.get('c1', 1e-4)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Wolfe':
            self.c1 = kwargs.get('c1', 1e-4)
            self.c2 = kwargs.get('c2', 0.9)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        else:
            raise ValueError('Unknown method {}'.format(self._method))

    
    def line_search(self, oracle, x_k, d_k, f_k, g_k, prev_alpha=None):
        phi_k = lambda a: oracle.func_directed(x_k, d_k, a)
        dphi_k = lambda a: oracle.grad_directed(x_k, d_k, a)
        phi_k.zero, dphi_k.zero = f_k, g_k @ d_k

        if self._method == 'Armijo':
            alpha = prev_alpha if prev_alpha else self.alpha_0
            while phi_k(alpha) > phi_k.zero + self.c1 * alpha * dphi_k.zero:
                alpha /= 2
            return alpha

        if self._method == 'Wolfe':
            alpha, *_ = scalar_search_wolfe2(phi=phi_k, derphi=dphi_k,
                                             phi0=phi_k.zero, derphi0=dphi_k.zero,
                                             c1=self.c1, c2=self.c2)
            if alpha:
                return alpha
            
            armijo = LineSearcher(method='Armijo', 
                                    c1=self.c1, alpha=self.alpha_0)
            return armijo.line_search(oracle, x_k, d_k, 
                                          f_k=f_k, g_k=g_k, prev_alpha=prev_alpha)


def SGD(oracle, beta_1=0.9, beta_2=0.999, **kwargs):
    line_search = kwargs.get('line_search', 'Armijo')
    iter_count = kwargs.get('iter_count', 100)
    optimizer = kwargs.get('optimizer', None)
    lr = kwargs.get('lr', None)
    
    searcher = LineSearcher(method=line_search)
    x = np.random.sample(oracle.n_features)

    t_start = perf_counter()
    oracle.next_batch()
    print(oracle.batch_size)
    func = oracle.func(x)
    grad = oracle.grad(x)
    grad0_norm = grad_norm = norm(grad)
    alpha = None
   
    i = 0
    m = v = 0
    for i in range(1, iter_count + 1):
        if optimizer == 'adam':
            if not lr:
                lr = 1e-3
                
            m = beta_1 * m + (1-beta_1) * grad
            v = beta_2 * v + (1-beta_2) * (grad ** 2)
            m_cap = m / (1 - (beta_1 ** i))
            v_cap = v / (1 - (beta_2 ** i))
            x = x - (lr * m_cap) / (v_cap ** 0.5 + 1e-8)
        else:
            #alpha = searcher.line_search(oracle, x_k=x, d_k=-grad,
            #                         f_k=func, g_k=grad,
            #                         prev_alpha=2 * alpha if alpha else None)
            alpha = 100 / ((i + 1) ** 0.6)
            x -= alpha * grad
        
        oracle.next_batch()
        func = oracle.func(x)
        grad = oracle.grad(x)
        grad_norm = norm(grad)
    
    t_end = perf_counter()
    result = {'f_opt': func,
              'grad_norm': grad_norm,
              'rk': grad_norm**2 / grad0_norm ** 2,
              'oracle_calls': {'f': oracle.f_calls, 
                               'df': oracle.g_calls, 
                               'ddf': oracle.h_calls},
              'solution': x.reshape(-1, 1).tolist(),
              'time': t_end - t_start}
    
    return result


def proximal_gradient(oracle, **kwargs):
    line_search = kwargs.get('line_search', 'Wolfe')
    iter_count = kwargs.get('iter_count', 1000)
    Lk = L0 = kwargs.get('L0', 1e-2)

    searcher = LineSearcher(method=line_search)
    x = np.random.sample(oracle.n_features)

    t_start = perf_counter()
    func = oracle.func(x)
    grad = oracle.grad(x)
    grad0_norm = grad_norm = norm(grad)
    
    for _ in range(iter_count): 
        while True:
            x_new = oracle.prox(x - grad / Lk, 1 / Lk)
            
            if oracle.func(x_new) > func + grad @ (x_new - x) + Lk / 2 * norm(x_new - x) ** 2:
                Lk *= 2
            else:
                alpha = searcher.line_search(oracle, x_k=x, d_k=-grad, f_k=func, g_k=grad)
                x += alpha * (x_new - x)
                break

        Lk /= 2
        func = oracle.func(x)
        grad = oracle.grad(x)
        grad_norm = norm(grad)
        
    t_end = perf_counter()
    result = {'f_opt': oracle.loss(x),
              'grad_norm': grad_norm,
              'rk': grad_norm**2 / grad0_norm ** 2,
              'oracle_calls': {'f': oracle.f_calls, 
                               'df': oracle.g_calls, 
                               'ddf': oracle.h_calls},
              'solution':x.reshape(-1, 1).tolist(),
              'time': t_end - t_start}
    
    return result



def solve_dataset(**kwargs):
    np.random.seed(kwargs['seed'])
    
    X, y = load_svmlight_file(kwargs['path'])
    X = X.tocsr()
    y_unique = np.unique(y)
    y = np.where(y == y_unique[0], 0, 1)
    oracle = LogRegOracle(X, y, regcoef=kwargs.get('regcoef', None),
                                batch_size=kwargs.get('batch_size', None))
    
    method = kwargs['method']
    if method == 'SGD':
        method = SGD
    elif method == 'proximal':
        method = proximal_gradient
    
    return method(oracle, **kwargs)


def parse_args():
    options = ['path=', 'method=', 'line_search=',
               'seed=', 'epsilon=', 'iter_count=',
               'regcoef=', 'batch_size=', 'optimizer=']
    try:
        opts, _ = getopt.getopt(sys.argv[1:], '', longopts=options)
    except getopt.GetoptError as e:
        print(e)
        print('Use valid options: ', options)
        sys.exit(2)

    kwargs = dict()
    kwargs['path'] = 'adult.txt'
    kwargs['method'] = 'SGD'
    kwargs['line_search'] = 'Wolfe'
    kwargs['seed'] = 42
    kwargs['epsilon'] = 1e-6
    kwargs['iter_count'] = 1000
    kwargs['regcoef'] = None
    kwargs['batch_size'] = None
    kwargs['optimizer'] = None 
    

    for opt, arg in opts:
        if opt == '--path':
            kwargs['path'] = arg
        elif opt == '--method':
            kwargs['method'] = arg
        elif opt == '--line_search':
            if arg == 'Lipshitz':
                print('Liphitz search method is not implemented...')
                sys.exit(2)
            kwargs['line_search'] = arg
        elif opt == '--seed':
            kwargs['seed'] = int(arg)
        elif opt == '--epsilon':
            kwargs['epsilon'] = float(arg)
        elif opt == '--iter_count':
            kwargs['iter_count'] = int(arg)
        elif opt == '--regcoef':
            kwargs['regcoef'] = float(arg)
        elif opt == '--batch_size':
            kwargs['batch_size'] = int(arg)
        elif opt == '--optimizer':
            kwargs['optimizer'] = 'adam'
            
    return kwargs


if __name__ == '__main__':
    kwargs = parse_args()
    solution = solve_dataset(**kwargs)
    with open('output.json', 'w') as f:
        json.dump(solution, f, indent=4, ensure_ascii=True)

