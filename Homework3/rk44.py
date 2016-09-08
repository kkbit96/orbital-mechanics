#!/usr/bin/python

from __future__ import division
import numpy as np

def compute_k(fun, ti, ri, vi, h):
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
    k.append(h*fun(ti+h/2, ri+(h/2)*vi+k[-2]/4, vi+k[-1]/2))
    k.append(h*fun(ti+h, ri+h*vi+k[-2]/2, vi+k[-1]))

    return k


def step(fun, ti, ri, vi, h):
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

    k = compute_k(fun, ti, ri, vi, h)

    # weights of k's for rk44
    wts = [1, 2, 2, 1]

    v = vi + np.dot(k, wts)/6

    r = ri + h*vi + (h/6)*sum(k[:-1])

    t = ti + h

    return t, r, v

def rk44(fun, ti, ri, vi, tf, h=0.01):
    """Performs Runge-Kutta integration for second order ODE.

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
    # check to see if ri and vi are iterable
    if hasattr(ri, '__iter__') and hasattr(vi, '__iter__'):
        t, r, v = ti, ri, vi
        isVector = True

    else:
        # create one-element iterable for loop
        t, r, v = ti, [ri], [vi]
        isVector = False

    # step until time reaches final time
    while t < tf:

        # define time step
        tstep = t

        # define dt for step (for last step < h)
        hstep = min(h, tf-t)

        # loop over all elements in r and v
        for i in xrange(len(r)):
            t, r[i], v[i] = step(fun, tstep, r[i], v[i], hstep)

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

    t, r, v = rk44(fun, ti=ti, ri=ri, vi=vi, tf=tf, h=h)

    print 'r = [{:.10f}  {:.10f}  {:.10f}]'.format(*r)
    print 'v = [{:.10f}  {:.10f}  {:.10f}]'.format(*v)
