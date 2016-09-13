#!/usr/bin/python

from __future__ import division
import numpy as np


def __compute_k(fun, t, v, h):
    """Computes the k values for the rk44 method.

    :fun: ODE function, must be in the form of f(t, v), return v
    :t: time for previous iteration
    :v: previous time step values (in iterable form)
    :h: time step
    :returns:
        :k: array of k values (two-dimensional numpy array)

    """
    # convert to numpy array
    v = np.array(v)

    # initialize k
    k = []

    # compute values for k from fun = f(ti, vi)
    k.append(h*np.array(fun(t, v)))
    k.append(h*np.array(fun(t+h/2, v+k[-1]/2)))
    k.append(h*np.array(fun(t+h/2, v+k[-1]/2)))
    k.append(h*np.array(fun(t+h, v+k[-1])))

    return np.array(k).T


def __step(fun, ti, vi, h):
    """Performs a single step of rk44 integration for second order ode.

    :fun: ODE function, must be in the form of f(t, v)
    :t: time at start of time step
    :v: values at start of time step (in iterable form)
    :h: time step
    :returns:
        :t: time at end of time step
        :v: values at end of time step (as list)

    """
    k = __compute_k(fun, ti, vi, h)

    # weights of k's for rk44
    wts = [1, 2, 2, 1]

    # compute velocity, use numpy's dot product to compute weighted sum
    v = vi + k.dot(wts)/6

    # increment time step
    t = ti + h

    return t, list(v)


def rk44(fun, ti, vi, tf, h=0.01):
    """Performs Runge-Kutta integration for second order ODE.

    :fun: ODE function, must be in the form of f(t, r, v)
    :ti: initial time
    :vi: initial conditions (in iterable form)
    :tf: final evaluation time
    :h: time step
    :returns:
        :t: final time value
        :v: final values at tf (as list)

    """

    # initialization of solution variables
    t = ti
    v = list(vi)

    # step until time reaches final time
    while t < tf:

        # allow for h to change to ensure time stops at tf (if necessary)
        hstep = min(h, tf-t)

        t, v = __step(fun, t, v, hstep)

    return t, v


if __name__ == "__main__":

    # second order ODE to solve
    def fun(t, vi):
        vi = np.array(vi)
        v = np.zeros(len(vi))
        v[:3] = vi[3:]
        v[3:] = 2*vi[3:] - vi[:3]

        return v

    ## initial conditions
    ti = 0.0
    vi = [2.0, 1.0, 3.0, 3.0, 2.0, 1.0]

    ## final conditions
    tf = 0.2
    h = 0.01

    #########################  rk44 solution #########################
    # call rk44 routine, store solution in t, r, v
    t, v = rk44(fun, ti=ti, vi=vi, tf=tf, h=h)

    print '\nRunge-Kutta solution (RK44), t = {}'.format(t)
    print 'r = [{:14.10f}  {:14.10f}  {:14.10f} ]'.format(*v[:3])
    print 'v = [{:14.10f}  {:14.10f}  {:14.10f} ]'.format(*v[3:])

    ######################### exact solution #########################
    ## coefficients
    # at ti = 0, c1 = ri
    c1 = np.array(vi[:3])

    # at ti = 0, c2 = vi - ri
    c2 = np.array(vi[3:]) - vi[:3]

    # computation of exact solution
    re = (c1 + c2*tf)*np.exp(tf)
    ve = ((c1 + c2) + c2*tf)*np.exp(tf)

    print '\nExact Solution, t = {}'.format(tf)
    print 'r = [{:14.10f}  {:14.10f}  {:14.10f} ]'.format(*re)
    print 'v = [{:14.10f}  {:14.10f}  {:14.10f} ]'.format(*ve)

    ######################## error calculation ########################
    rerr = v[:3] - re
    verr = v[3:] - ve

    print '\nRunge-Kutta error (RK44 - Exact)'
    print 'r = [{:14.10g}  {:14.10g}  {:14.10g} ]'.format(*rerr)
    print 'v = [{:14.10g}  {:14.10g}  {:14.10g} ]'.format(*verr)
