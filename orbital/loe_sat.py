#!/usr/bin/python

import numpy as np
        return v
def update_orbit(Ri, Vi, ti, tf, h, output=None):
    """Updates Keplerian orbit position

    :Ri: Position vector at time ti
    :Vi: Velocity vector at time ti
    :ti: Time for initial position and velocity vectors
    :tf: Computation end time
    :h: Time step
    :returns:
        :R: Position vector at time tf
        :V: Velocity vector at time tf

    """
    from numpy.linalg import norm
    from kepler import mu
    from rk44 import rk44, __step

    # equation for acceleration: d2r/dt2 = -(mu/|r|^3)*r
    def accel(t, vi):

        # magnitude of position vector
        r = norm(vi[:3])

        # initialize solution vector
        v = np.zeros(len(vi))

        # dr/dt = v
        v[:3] = vi[3:]

        # d2r/dt2 = -(mu/|r|^3)*r
        v[3:] = -(mu/r**3)*vi[:3]

        return v

    # combine position and velocity vectors for RK44 routine
    vi = list(Ri) + list(Vi)

    t, v = ti, list(vi)

    if output is not None:
        with open('loe_sat.dat', 'w') as writer:

            # write headers to file
            writer.write('     a                 e             i'
                         + '           Omega         omega         nu\n')
            writer.write('{:10.10f}{:14.10f}{:14.10f}{:14.10f}{:14.10f}'
                         + '{:14.10f}\n'.format(*elements0))

            # step until time reaches final time
            while t < tf:

                # allow for h to change to ensure time stops at tf
                # (if necessary)
                hstep = min(h, tf-t)

                # perform Runge-Kutta integration step
                t, v = __step(fun=accel, ti=t, vi=v, h=hstep)

                # find elements from final position and velocity vectors
                elements = rvToElements(v[:3], v[3:])

                # write orbital elements for time step to file
                writer.write('{:10.10f}{:14.10f}{:14.10f}{:14.10f}'
                             + '{:14.10f}{:14.10f}\n'.format(*elements))

    else:

        # integration
        t, v = rk44(fun=accel, ti=ti, vi=vi, tf=tf, h=h)

    return t, v[:3], v[3:]
        return t, v[:3], v[3:]


if __name__ == "__main__":
    from kepler import rvToElements, elementsToRV
    from numpy.linalg import norm

    names = ['a', 'e', 'i', 'Omega', 'omega', 'nu']

    # initial orbital elements
    dt = 15.0       # sec
    a0 = 6652.55563 # km
    e0 = 0.025
    i0 = 40.0       # deg
    Omega0 = 20.0   # deg
    omega0 = 30.0   # deg
    nu0 = 50.0      # deg

    elements0 = [a0, e0, i0, Omega0, omega0, nu0]

    # initial position, velocity vectors
    R0, V0 = elementsToRV(a=a0, e=e0, i=i0, Omega=Omega0,
                          omega=omega0, nu=nu0)

    t0 = 0.0
    tf = 24*3600

    #################### Orbit Determination Error  ####################
    # compute new orbit position and velocity for final time
    t, Rf, Vf = update_orbit(Ri=R0, Vi=V0, ti=t0, tf=tf, h=dt,
                             output='loe_sat.dat')

    # compute error values position and velocity vectors
    Rerr = Rf - R0
    Verr = Vf - V0

    # find elements from final position and velocity vectors
    elements = rvToElements(Rf, Vf)

    # compute error values for orbital elements
    errs = [el1 - el for el1, el in zip(elements, elements0)]

    print '\n     Position and Velocity Vectors'
    print '=========================================='

    print '\nInitial Conditions'
    print ' {:s}:  [ {:14.10f}  {:14.10f}  {:14.10f} ]'.format('R', *R0)
    print ' {:s}:  [ {:14.10f}  {:14.10f}  {:14.10f} ]'.format('V', *V0)

    print '\nFinal Conditions'
    print ' {:s}:  [ {:14.10f}  {:14.10f}  {:14.10f} ]'.format('R', *Rf)
    print ' {:s}:  [ {:14.10f}  {:14.10f}  {:14.10f} ]'.format('V', *Vf)

    print '\nDifference'
    print ' {:s}:  [ {:14.10g}  {:14.10g}  {:14.10g} ]'.format('R', *Rerr)
    print ' {:s}:  [ {:14.10g}  {:14.10g}  {:14.10g} ]'.format('V', *Verr)

    print '\nAbsolute Difference'
    print ' {:s}:  {:14.10g}'.format('R', norm(Rf) - norm(R0))
    print ' {:s}:  {:14.10g}'.format('V', norm(Vf) - norm(V0))

    print '\n          Orbital Elements'
    print '=========================================='

    print '         Initial Condition       Final Condition',
    print '         Difference'
    print '        -------------------     -----------------',
    print '     -----------------'
    for el, el1, el2, err in zip(names, elements0, elements, errs):
        print '{:5s}     {:16.10f}      {:16.10f}      {:16.10g}' \
            .format(el, el1, el2, err)
