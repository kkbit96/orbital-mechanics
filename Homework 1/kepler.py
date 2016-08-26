#!/usr/bin/python

import numpy as np


# Gravitational parameter
mu = 3.98600436e5  # km^3/s^2


def rvToElements(Rvec, Vvec):
    """Computes the Keplerian orbital elements of an Earth satellite
    using the geocentric-equatorial position and velocity vectors
    of the orbiting body.

    :Rvec:  position vector to orbiting body
    :Vvec:  velocity vector of orbiting body
    :returns:
    :a:     semi-major axis
    :e:     eccentricity
    :i:     inclination
    :Omega: longitude of ascending node
    :omega: argument of periapsis
    :nu:    true anomaly
    """
    from numpy.linalg import norm

    def testAngle(test, angle):
        """Checks test for sign and returns corrected angle"""
        angle *= 180./np.pi
        if test > 0:
            return angle
        elif test < 0:
            return 360. - angle

    Rvec = np.array(Rvec)
    Vvec = np.array(Vvec)

    # Computation of e (eccentricity)
    R = norm(Rvec)
    V = norm(Vvec)

    eVec = (V**2/mu-1/R)*Rvec - Rvec.dot(Vvec)*Vvec/mu
    e = norm(eVec)

    # Computation of a (semi-major axis)
    hVec = np.cross(Rvec, Vvec)
    h = norm(hVec)
    p = h**2/mu
    a = p/(1-e**2)

    # Computation of i (inclination)
    i = np.arccos(hVec[2]/h)*180./np.pi

    # Computation of Omega (longitude of ascending node)
    Nvec = np.array([-hVec[1], hVec[0], 0])
    N = norm(Nvec)
    OmegaPi = np.arccos(-hVec[1]/N)
    Omega = testAngle(Nvec[1], OmegaPi)

    # Computation of omega (argument of periapsis)
    omegaPi = np.arccos(Nvec.dot(eVec)/(N*e))
    omega = testAngle(eVec[2], omegaPi)

    # Computation of nu (true anomaly)
    nuPi = np.arccos(eVec.dot(Rvec)/(e*R))
    nu = testAngle(Rvec.dot(Vvec), nuPi)

    return a, e, i, Omega, omega, nu


def elementsToRV(a, e, i, Omega, omega, nu):
    """Computes the geocentric-equatorial position and velocity
    vectors of an Earth satellite using the Keplerian orbital
    elements.

    :a:     semi-major axis
    :e:     eccentricity
    :i:     inclination
    :Omega: longitude of ascending node
    :omega: argument of periapsis
    :nu:    true anomaly
    :returns:
    :Rvec:  position vector to orbiting body
    :Vvec:  velocity vector of orbiting body
    """
    from numpy import sin, cos

    def transformationMatrix(i, Omega, omega):
        """Generates matrix for transformation from perifocal coordinate
        system to geocentric-equatorial coordinate system.
        Note: All angles MUST be in RADIANS

        :i:     inclination
        :Omega: longitude of ascending node
        :omega: argument of periapsis
        :returns: Transformation Matrix

        """
        T = np.zeros((3, 3))

        T[0, 0] =  cos(Omega)*cos(omega) - sin(Omega)*sin(omega)*cos(i)
        T[0, 1] = -cos(Omega)*sin(omega) - sin(Omega)*cos(omega)*cos(i)
        T[0, 2] =  sin(Omega)*sin(i)

        T[1, 0] =  sin(Omega)*cos(omega) + cos(Omega)*sin(omega)*cos(i)
        T[1, 1] = -sin(Omega)*sin(omega) + cos(Omega)*cos(omega)*cos(i)
        T[1, 2] = -cos(Omega)*sin(i)

        T[2, 0] =  sin(omega)*sin(i)
        T[2, 1] =  cos(omega)*sin(i)
        T[2, 2] =  cos(i)

        return T

    # Conversion of angles from degrees to radians
    i *= np.pi/180.
    Omega *= np.pi/180.
    omega *= np.pi/180.
    nu *= np.pi/180.

    p = a*(1 - e**2)
    r = p/(1 + e*cos(nu))
    rVec = r*np.array([cos(nu), sin(nu), 0])
    vVec = np.sqrt(mu/p)*np.array([-sin(nu), e + cos(nu), 0])

    T = transformationMatrix(i, Omega, omega)

    Rvec = T.dot(rVec)
    Vvec = T.dot(vVec)

    return Rvec, Vvec


def main():
    # First problem
    print '                  First Problem'
    print ' =================================================='
    # First case
    R = [-1020.9884199769, 4110.9705239038,  5931.5781229624]    # km
    V = [-5.71571324, -4.50160713, 2.57194489]                   # km/sec
    a1, e1, i1, Omega1, omega1, nu1 = rvToElements(R, V)

    # Second case
    R = [8948.76740168, -3214.34029976, -663.27465526]   # km
    V = [-2.27456408, - 5.99222926, -0.77384874]         # km/sec
    a2, e2, i2, Omega2, omega2, nu2 = rvToElements(R, V)

    print ' elem        First Case               Second Case'
    print ' -----    -----------------        -----------------'
    print ' {:5s}    {:13.8f} km         {:13.8f} km'.format('a', a1, a2)
    print ' {:5s}    {:13.8f}            {:13.8f}'.format('e', e1, e2)
    print ' {:5s}    {:13.8f} deg        {:13.8f} deg'.format('i', i1, i2)
    print ' {:5s}    {:13.8f} deg        {:13.8f} deg'.format('Omega', Omega1, Omega2)
    print ' {:5s}    {:13.8f} deg        {:13.8f} deg'.format('omega', omega1, omega2)
    print ' {:5s}    {:13.8f} deg        {:13.8f} deg'.format('nu', nu1, nu2)

    # Second problem
    print '\n Second Problem'
    print ' ============='
    # First case
    a = 9000.0      # km
    e = 0.050
    i = 40.0        # deg
    Omega = 45.0    # deg
    omega = 50.0    # deg
    nu = 20.0       # deg
    R1, V1 = elementsToRV(a, e, i, Omega, omega, nu)

    # Second case
    a = 9400.0      # km
    e = 0.015
    i = 164.0       # deg
    Omega = 199.0   # deg
    omega = 310.0   # deg
    nu = 206.0      # deg
    elementsToRV(a, e, i, Omega, omega, nu)
    R2, V2 = elementsToRV(a, e, i, Omega, omega, nu)
    print ' First Case'
    print ' ----------'
    print ' {:s}:  [ {:14.8f}  {:14.8f}  {:14.8f} ]'.format('R', *R1)
    print ' {:s}:  [ {:14.8f}  {:14.8f}  {:14.8f} ]'.format('V', *V1)

    print '\n Second Case'
    print ' -----------'
    print ' {:s}:  [ {:14.8f}  {:14.8f}  {:14.8f} ]'.format('R', *R2)
    print ' {:s}:  [ {:14.8f}  {:14.8f}  {:14.8f} ]'.format('V', *V2)


def test():
    # First problem
    print 'Error in First Problem'
    print '======================'
    # First case
    R1 = [-1020.9884199769, 4110.9705239038,  5931.5781229624]    # km
    V1 = [-5.71571324, -4.50160713, 2.57194489]                   # km/sec
    R1b, V1b = elementsToRV(*rvToElements(R1, V1))

    # Second case
    R2 = [8948.76740168, -3214.34029976, -663.27465526]   # km
    V2 = [-2.27456408, - 5.99222926, -0.77384874]         # km/sec
    R2b, V2b = elementsToRV(*rvToElements(R2, V2))
    print 'First Case'
    print '----------'
    print '{:s}:  [ {:14.8g}  {:14.8g}  {:14.8g} ]'.format('R', *(R1b - R1))
    print '{:s}:  [ {:14.8g}  {:14.8g}  {:14.8g} ]'.format('V', *(V1b - V1))

    print '\nSecond Case'
    print '-----------'
    print '{:s}:  [ {:14.8g}  {:14.8g}  {:14.8g} ]'.format('R', *(R2b - R2))
    print '{:s}:  [ {:14.8g}  {:14.8g}  {:14.8g} ]'.format('V', *(V2b - V2))

    # Second problem
    print '\n            Error in Second Problem'
    print '            ======================='
    # First case
    a1 = 9000.0      # km
    e1 = 0.050
    i1 = 40.0        # deg
    Omega1 = 45.0    # deg
    omega1 = 50.0    # deg
    nu1 = 20.0       # deg
    a1b, e1b, i1b, Omega1b, omega1b, nu1b = rvToElements(*elementsToRV(a1, e1, i1, Omega1, omega1, nu1))

    # Second case
    a2 = 9400.0      # km
    e2 = 0.015
    i2 = 164.0       # deg
    Omega2 = 199.0   # deg
    omega2 = 310.0   # deg
    nu2 = 206.0      # deg
    a2b, e2b, i2b, Omega2b, omega2b, nu2b = rvToElements(*elementsToRV(a2, e2, i2, Omega2, omega2, nu2))
    print '          First Case         Second Case'
    print '        --------------     --------------'
    print '{:5s}   {:14.8g}      {:14.8g}'.format('a', a1b-a1, a2b-a2)
    print '{:5s}   {:14.8g}      {:14.8g}'.format('e', e1b-e1, e2b-e2)
    print '{:5s}   {:14.8g}      {:14.8g}'.format('i', i1b-i1, i2b-i2)
    print '{:5s}   {:14.8g}      {:14.8g}'.format('Omega', Omega1b-Omega1, Omega2b-Omega2)
    print '{:5s}   {:14.8g}      {:14.8g}'.format('omega', omega1b-omega1, omega2b-omega2)
    print '{:5s}   {:14.8g}      {:14.8g}'.format('nu', nu1b-nu1, nu2b-nu2)

if __name__ == "__main__":
    main()
    print '\n\n'
    test()
