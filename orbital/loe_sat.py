#!/usr/bin/python

import numpy as np
from numpy.linalg import norm

import rk44
import kepler
from kepler import mu, re, rvToElements

J = (-1.0,                      # J0
      0.0,                      # J1
      0.1082635666551098e-2,    # J2
     -0.2532473691332948e-5,    # J3
     -0.1619974305782220e-5,    # J4
     )

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
        self.n = [0]
        self.__fun__ = self.__accel__


    def set_accel(self, fun):
        """Sets function for acceleration to be used in RK44 routine

        :fun: function to use for acceleration, must take t, vi and deg as arguments
        """
        self.__fun__ = fun

    def setPertDegree(self, n=[0]):
        """Sets the list of degrees for the perturbations for the orbit acceleration.

        :n: List of degrees for perturbations

        """
        self.n = n

    def __write_output__(self, writer):
        """Writes orbital elements to file output
        """

        elements = self.t, self.a, self.e, self.i, self.Omega, self.omega, self.nu

        # write headers to file
        writer.write(('{:8.0f}{:18.10f}{:16.10f}{:16.10f}{:16.10f}{:16.10f}'
                    + '{:16.10f}\n').format(*elements))

    @staticmethod
    def __accel__(t, vi, deg=[0]):
        """Equation for acceleration due to gravity.

        :t: current time
        :vi: current position and velocity vector
        :n: degrees of perturbation terms to use ([0] = 2-body)

        """
        from numpy.polynomial.legendre import legval, legder

        # initialize solution vector
        v = np.zeros(len(vi))

        # dr/dt = v
        v[:3] = vi[3:]

        # magnitude of position vector
        r = norm(vi[:3])

        # sin(phi) = z/r
        sinphi = vi[2]/r

        xi = re/r

        # Common coefficients for Legendre polynomials and derivatives
        mult = [J[n]*xi**n if n in deg else 0 for n in xrange(max(deg)+1)]

        # compute coefficients for Legendre polynomial evaluation
        cPn = (np.arange(max(deg)+1)+1)*mult

        # Legendre summation
        Pn = legval(sinphi, cPn)

        # Legendre derivative summation
        Pn_prime = legval(sinphi, legder(mult))

        # common acceleration
        v[3:] = (mu/r**3)*(Pn + sinphi*Pn_prime)*vi[:3]

        # additive term for z-acceleration
        v[-1] -= (mu/r**3)*(r*Pn_prime)

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
            self.t, v = rk44.rk44(fun=self.__fun__, ti=self.t, vi=vi, tf=tf, h=h)

            self.R, self.V = np.array(v[:3]), np.array(v[3:])

        else:
            with open(self.output, 'w') as writer:

                # write headers to file
                writer.write('       t          a                 e               '
                             'i             Omega          omega             nu\n')

                # write initial orbital elements
                self.__write_output__(writer)

                # step until time reaches final time
                while self.t < tf:

                    # allow for h to change to ensure time stops at tf (if necessary)
                    hstep = min(h, tf-self.t)

                    # perform Runge-Kutta integration step
                    self.update_orbit(hstep)

                    # write orbital elements for time step to file
                    self.__write_output__(writer)

    def update_orbit(self, h):
        """Updates Keplerian orbit position

        :h: Time step

        """
        # combine position and velocity vectors for RK44 routine
        vi = list(self.R) + list(self.V)

        t, v = self.t, list(vi)

        # perform Runge-Kutta integration step
        self.t, v = rk44.step(self.__fun__, t, v, h, self.n)

        self.R, self.V = np.array(v[:3]), np.array(v[3:])

        # find elements from final position and velocity vectors
        self.a, self.e, self.i, self.Omega, self.omega, self.nu \
            = rvToElements(self.R, self.V)


def main(elements0, tf, dt, deg=(0,)):
    from kepler import elementsToRV

    names = ['a', 'e', 'i', 'Omega', 'omega', 'nu']

    if len(deg) == 1:
        _file = 'loe_sat{}.dat'.format(deg[0])
    else:
        _file = 'loe_sat{}.dat'.format(deg)

    loe_sat = LOE_Satellite(*elements0, output=_file)

    # initial position, velocity vectors
    R0, V0 = elementsToRV(*elements0)

    # set perturbation to two-body
    if len(deg) == 1 and deg[0] == 0:
        print '\n\n============================== Two Body =============================='
    elif 0 in deg:
        print '\n\n============================== Include',
        print ', '.join(['J{}'.format(d) for d in deg if d != 0]),
        print '=============================='
    else:
        print '\n\n============================== Only',
        print ', '.join(['J{}'.format(d) for d in deg if d != 0]),
        print '=============================='

    # set perturbations for orbit
    loe_sat.setPertDegree(deg)

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


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    elements = 'a', 'e', 'i', 'Omega', 'omega', 'nu'

    dt = 15           # sec
    tf = 7*24*3600    # sec

    a0 = 6652.555663  # km
    e0 = 0.05
    i0 = 20.0         # deg
    Omega0 = 30.0     # deg
    omega0 = 40.0     # deg
    nu0 = 50.0        # deg

    elements0 = a0, e0, i0, Omega0, omega0, nu0

    # perturbations = (0,), (0,2), (0,3), (0,4)
    perturbations = (0,2), (0,3), (0,4)

    for pert_orders in perturbations:
        main(elements0, tf, dt, pert_orders)

    data = {}

    for pert_orders in perturbations:
        if len(pert_orders) == 1:
            _file = 'loe_sat{}.dat'.format(pert_orders[0])
        else:
            _file = 'loe_sat{}.dat'.format(pert_orders)
        datum = {}

        with open(_file, 'r') as datafile:

            for j, line in enumerate(datafile):

                if j == 0:
                    hdr = line.split()
                    for e in hdr:
                        datum[e] = []

                else:
                    dataline = line.split()
                    for k, e in enumerate(hdr):
                        datum[e].append(float(dataline[k]))

        if len(pert_orders) == 1:
            data[str(pert_orders[0])] = datum
        else:
            data[str(pert_orders)] = datum

    twobody = data['0']
    time = [t/3600 for t in twobody['t']]

    for key, datum in sorted(data.iteritems()):
        if key != '0':
            fig, ax = plt.subplots(3, 2, sharex=True)

            for i, elem in enumerate(elements):
                j = i % 3
                k = i / 3
                plt.sca(ax[j][k])
                # element = [a-a2 for a, a2 in zip(datum[elem], twobody[elem])]
                # element = np.unwrap([a-a2 for a, a2 in zip(datum[elem], twobody[elem])])

                # element = np.unwrap(datum[elem])
                # plt.plot(time, element, label=key)

                plt.plot(time, datum[elem], label=key)
                # plt.plot(time[:1000], element[:1000], label='J{}'.format(i))
                # plt.plot(time[-1000:], element[-1000:], label='J{}'.format(i))

                if j == 2:
                    plt.xlabel('t [hr]')
                plt.ylabel(elem)

            plt.suptitle(key)
            plt.show()
