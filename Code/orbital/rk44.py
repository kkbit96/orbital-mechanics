#!/usr/bin/python

from __future__ import division
import numpy as np


def __compute_k(fun, ti, ri, vi, h):
    """Computes the k values for the rk44 method.

    :fun: function for second order ODE
    :ti: time for previous iteration
    :ri: previous position
    :vi: previous velocity
    :h: time step
    :returns:
        :k: array of k values

    """
    # initialize k
    k = []

    # compute values for k from fun = f(ti, ri, vi)
    k.append(h*fun(ti, ri, vi))
    k.append(h*fun(ti+h/2, ri+(h/2)*vi, vi+k[-1]/2))
    k.append(h*fun(ti+h/2, ri+(h/2)*vi+(h*k[-2]/4), vi+k[-1]/2))
    k.append(h*fun(ti+h, ri+h*vi+(h*k[-2]/2), vi+k[-1]))

    return k


def __step(fun, ti, ri, vi, h):
    """Performs a single step of rk44 integration for second order ode.

    :fun: ODE function, must be in the form of f(t, r, v)
    :ti: initial time
    :ri: position initial conditions
    :vi: velocity initial conditions
    :tf: final evaluation time
    :h: time step
    :returns:
        :r: final position at tf
        :v: final velocity at tf

    """
    k = __compute_k(fun, ti, ri, vi, h)

    # weights of k's for rk44
    wts = [1, 2, 2, 1]

    # compute velocity, use numpy's dot product to compute weighted sum
    v = vi + np.dot(k, wts)/6

    # compute position
    r = ri + h*vi + (h/6)*sum(k[:-1])

    return r, v


def rk44(fun, ti, ri, vi, tf, h=0.01):
    """Performs Runge-Kutta integration for second order ODE.

    :fun: ODE function, must be in the form of f(t, r, v)
    :ti: initial time
    :ri: position initial conditions
    :vi: velocity initial conditions
    :tf: final evaluation time
    :h: time step
    :returns:
        :t: final time value
        :r: final position at tf
        :v: final velocity at tf

    """
    # check to see if ri and vi are iterable
    if hasattr(ri, '__iter__') and hasattr(vi, '__iter__'):
        # initialize t, r, v with initial conditions
        t, r, v = ti, list(ri), list(vi)
        isVector = True

    else:
        # initialize t, r, v with initial conditions, creating
        #   one-element iterables for r, v for loop structure
        t, r, v = ti, [ri], [vi]
        isVector = False

    # step until time reaches final time
    while t < tf:

        # allow for h to change to ensure time stops at tf (if necessary)
        hstep = min(h, tf-t)

        # loop over all elements in r and v
        for i in xrange(len(r)):
            r[i], v[i] = __step(fun, t, r[i], v[i], hstep)

        # increment time step
        t += h

    if isVector:
        return t, r, v

    return t, r[0], v[0]


if __name__ == "__main__":

    # second order ODE to solve
    fun = lambda t, r, v: 2*v - r

    ## initial conditions
    ti = 0.0
    ri = [2.0, 1.0, 3.0]
    vi = [3.0, 2.0, 1.0]

    ## final conditions
    tf = 0.2
    h = 0.01

    #########################  rk44 solution #########################
    # call rk44 routine, store solution in t, r, v
    t, r, v = rk44(fun, ti=ti, ri=ri, vi=vi, tf=tf, h=h)

    print '\nRunge-Kutta solution (RK44), t = {}'.format(t)
    print 'r = [{:14.10f}  {:14.10f}  {:14.10f} ]'.format(*r)
    print 'v = [{:14.10f}  {:14.10f}  {:14.10f} ]'.format(*v)

    ######################### exact solution #########################
    ## coefficients
    # at ti = 0, c1 = ri
    c1 = np.array(ri)

    # at ti = 0, c2 = vi - ri
    c2 = np.array(vi) - ri

    # computation of exact solution
    re = (c1 + c2*tf)*np.exp(tf)
    ve = ((c1 + c2) + c2*tf)*np.exp(tf)

    print '\nExact Solution, t = {}'.format(tf)
    print 'r = [{:14.10f}  {:14.10f}  {:14.10f} ]'.format(*re)
    print 'v = [{:14.10f}  {:14.10f}  {:14.10f} ]'.format(*ve)

    ######################## error calculation ########################
    rerr = r - re
    verr = v - ve

    print '\nRunge-Kutta error (RK44 - Exact)'
    print 'r = [{:14.10g}  {:14.10g}  {:14.10g} ]'.format(*rerr)
    print 'v = [{:14.10g}  {:14.10g}  {:14.10g} ]'.format(*verr)
