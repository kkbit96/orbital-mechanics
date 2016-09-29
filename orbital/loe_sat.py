#!/usr/bin/python

import numpy as np

import rk44
import kepler
from kepler import mu

class LOE_Satellite(object):

    """LOE_Satellite is a class that maintains the current position and
    velocity vectors as well as the orbital elements for a low earth satellite."""

    def __init__(self, a, e, i, Omega, omega, nu, ti=0, output=None):
        """Initializes satellite wit provided orbital elements. """
        self.a = a
        self.e = e
        self.i = i
        self.Omega = Omega
        self.omega = omega
        self.nu = nu
        self.t = ti
        self.output = output
        self.R, self.V = kepler.elementsToRV(a, e, i, Omega, omega, nu)

    def __write_output(self, writer):
        """Writes orbital elements to file output
        """

        elements = self.a, self.e, self.i, self.Omega, self.omega, self.nu

        # write headers to file
        writer.write('     a                 e             i'
                    + '           Omega         omega         nu\n')
        writer.write('{:10.10f}{:14.10f}{:14.10f}{:14.10f}{:14.10f}'
                    + '{:14.10f}\n'.format(*elements))

    @staticmethod
    def __accel(t, vi):
        """Equation for acceleration: d2r/dt2 = -(mu/|r|^3)*r

        """

        # magnitude of position vector
        r = norm(vi[:3])

        # initialize solution vector
        v = np.zeros(len(vi))

        # dr/dt = v
        v[:3] = vi[3:]

        # d2r/dt2 = -(mu/|r|^3)*r
        v[3:] = -(mu/r**3)*vi[:3]

        return v

    def run_orbit(self, tf, h):
        """Updates Keplerian orbit position

        :tf: Computation end time
        :h: Time step

        """

        if self.output is None:

            # combine position and velocity vectors for RK44 routine
            vi = list(self.R) + list(self.V)

            # integration
            self.t, v = rk44.rk44(fun=self.__accel, ti=self.t, vi=vi, tf=tf, h=h)

            self.R, self.V = np.array(v[:3]), np.array(v[3:])

        else:
            with open(self.output, 'w') as writer:

                # write headers to file
                writer.write('     a                 e             i           Omega         omega         nu\n')

                # write initial orbital elements
                self.__write_output(writer)

                # step until time reaches final time
                while self.t < tf:

                    # allow for h to change to ensure time stops at tf (if necessary)
                    hstep = min(h, tf-self.t)

                    self.update_orbit(h)
                    # perform Runge-Kutta integration step
                    self.update_orbit(hstep)

                    # write orbital elements for time step to file
                    self.__write_output(writer)

    def update_orbit(self, h):
        """Updates Keplerian orbit position
        :h: Time step

        """
        # combine position and velocity vectors for RK44 routine
        vi = list(self.R) + list(self.V)

        t, v = self.t, list(vi)

        # perform Runge-Kutta integration step
        self.t, v = rk44.step(fun=self.__accel, ti=t, vi=v, h=h)

        self.R, self.V = np.array(v[:3]), np.array(v[3:])

        # find elements from final position and velocity vectors
        self.a, self.e, self.i, self.Omega, self.omega, self.nu \
            = rvToElements(self.R, self.V)


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

    loe_sat = LOE_Satellite(*elements0, output='loe_sat.dat')

    # initial position, velocity vectors
    R0, V0 = elementsToRV(a=a0, e=e0, i=i0, Omega=Omega0,
                          omega=omega0, nu=nu0)

    t0 = 0.0
    tf = 24*3600

    #################### Orbit Determination Error  ####################
    # compute new orbit position and velocity for final time
    loe_sat.run_orbit(tf=tf, h=dt)

    # compute error values position and velocity vectors
    Rerr = loe_sat.R - R0
    Verr = loe_sat.V - V0

    # find elements from final position and velocity vectors
    elements = loe_sat.a, loe_sat.e, loe_sat.i, \
        loe_sat.Omega, loe_sat.omega, loe_sat.nu

    # compute error values for orbital elements
    errs = [el1 - el for el1, el in zip(elements, elements0)]

    print '\n     Position and Velocity Vectors'
    print '=========================================='

    print '\nInitial Conditions'
    print ' {:s}:  [ {:14.10f}  {:14.10f}  {:14.10f} ]'.format('R', *R0)
    print ' {:s}:  [ {:14.10f}  {:14.10f}  {:14.10f} ]'.format('V', *V0)

    print '\nFinal Conditions'
    print ' {:s}:  [ {:14.10f}  {:14.10f}  {:14.10f} ]'.format('R', *loe_sat.R)
    print ' {:s}:  [ {:14.10f}  {:14.10f}  {:14.10f} ]'.format('V', *loe_sat.V)

    print '\nDifference'
    print ' {:s}:  [ {:14.10g}  {:14.10g}  {:14.10g} ]'.format('R', *Rerr)
    print ' {:s}:  [ {:14.10g}  {:14.10g}  {:14.10g} ]'.format('V', *Verr)

    print '\nAbsolute Difference'
    print ' {:s}:  {:14.10g}'.format('R', norm(loe_sat.R) - norm(R0))
    print ' {:s}:  {:14.10g}'.format('V', norm(loe_sat.V) - norm(V0))

    print '\n          Orbital Elements'
    print '=========================================='

    print '         Initial Condition       Final Condition',
    print '         Difference'
    print '        -------------------     -----------------',
    print '     -----------------'
    for el, el1, el2, err in zip(names, elements0, elements, errs):
        print '{:5s}     {:16.10f}      {:16.10f}      {:16.10g}' \
            .format(el, el1, el2, err)
