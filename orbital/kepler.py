#!/usr/bin/python

import numpy as np

# Gravitational parameter
mu_earth = 3.98600436e5   # km^3/s^2
mu_sun = 1.3271240018e11  # km^3/s^2
re = 6378.137       # km


def julian_date(year, month, day, hour, minute, seconds):
    """Computes Julian date for specified date vector.

    :date_time: list or tuple with year, month, day, hour, minute
    :returns: Julian date number.

    """
    j0 = 367*year - np.floor(7*(year + np.floor((month+9)/12))/4) + np.floor(275*month/9) + day + 1721013.5
    jd = j0 + hour/24. + (minute+seconds/60.)/1440.

    return jd


def __test_angle__(test, angle):
    """Checks test for sign and returns corrected angle"""
    angle = np.rad2deg(angle)

    if test > 0:
        return angle
    elif test < 0:
        return 360. - angle


def rvToElements(Rvec, Vvec, mu=mu_earth):
    """Computes the Keplerian orbital elements of an Earth
    satellite using the geocentric-equatorial position and
    velocity vectors of the orbiting body.

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
    i = np.rad2deg(np.arccos(hVec[2]/h))

    # Computation of Omega (longitude of ascending node)
    Nvec = np.array([-hVec[1], hVec[0], 0])
    N = norm(Nvec)
    OmegaPi = np.arccos(-hVec[1]/N)
    Omega = __test_angle__(Nvec[1], OmegaPi)

    # Computation of omega (argument of periapsis)
    omegaPi = np.arccos(Nvec.dot(eVec)/(N*e))
    omega = __test_angle__(eVec[2], omegaPi)

    # Computation of nu (true anomaly)
    nuPi = np.arccos(eVec.dot(Rvec)/(e*R))
    nu = __test_angle__(Rvec.dot(Vvec), nuPi)

    return a, e, i, Omega, omega, nu


def __transformationMatrix__(i, Omega, omega):
    """Generates matrix for transformation from perifocal
    coordinate system to geocentric-equatorial coordinate
    system.
    Note: All angles MUST be in DEGREES

    :i:     inclination
    :Omega: longitude of ascending node
    :omega: argument of periapsis
    :returns: Transformation Matrix

    """
    from numpy import sin, cos

    i = np.deg2rad(i)
    Omega = np.deg2rad(Omega)
    omega = np.deg2rad(omega)

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


def elementsToRV(a, e, i, Omega, omega, nu, mu=mu_earth):
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
    # Conversion of nu from degrees to radians
    nu = np.deg2rad(nu)

    p = a*(1 - e**2)
    r = p/(1 + e*np.cos(nu))
    rVec = r*np.array([np.cos(nu), np.sin(nu), 0])
    vVec = np.sqrt(mu/p)*np.array([-np.sin(nu), e + np.cos(nu), 0])

    T = __transformationMatrix__(i, Omega, omega)

    Rvec = T.dot(rVec)
    Vvec = T.dot(vVec)

    return Rvec, Vvec


def __main__():
    names = ['a', 'e', 'i', 'Omega', 'omega', 'nu']

    # First problem
    print '                  First Problem'
    print ' =================================================='
    # First case
    R = [-1020.9884199769, 4110.9705239038,  5931.5781229624]    # km
    V = [-5.71571324, -4.50160713, 2.57194489]                   # km/sec
    elements1 = rvToElements(R, V)

    # Second case
    R = [8948.76740168, -3214.34029976, -663.27465526]   # km
    V = [-2.27456408, - 5.99222926, -0.77384874]         # km/sec
    elements2 = rvToElements(R, V)

    print ' elem        First Case               Second Case'
    print ' -----    -----------------      -----------------'

    units = ['km', ''] + 4*['deg']
    for outputs in zip(names, elements1, units, elements2, units):
        print ' {:5s}{:17.8f} {:3s}{:20.8f} {:3s}'.format(*outputs)

    # Second problem
    print '\n                     Second Problem'
    print ' ======================================================'
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
    R2, V2 = elementsToRV(a, e, i, Omega, omega, nu)

    print ' First Case'
    print ' ----------'
    print ' {:s}:  [ {:14.8f}  {:14.8f}  {:14.8f} ]'.format('R', *R1)
    print ' {:s}:  [ {:14.8f}  {:14.8f}  {:14.8f} ]'.format('V', *V1)

    print '\n Second Case'
    print ' -----------'
    print ' {:s}:  [ {:14.8f}  {:14.8f}  {:14.8f} ]'.format('R', *R2)
    print ' {:s}:  [ {:14.8f}  {:14.8f}  {:14.8f} ]'.format('V', *V2)


def __test__():
    names = ['a', 'e', 'i', 'Omega', 'omega', 'nu']

    # First problem
    print '            Error Check for First Problem'
    print '======================================================'

    # First case
    R = [-1020.9884199769, 4110.9705239038,  5931.5781229624]    # km
    V = [-5.71571324, -4.50160713, 2.57194489]                   # km/sec
    R1, V1 = elementsToRV(*rvToElements(R, V))
    Rerr1 = R1 - R
    Verr1 = V1 - V

    # Second case
    R = [8948.76740168, -3214.34029976, -663.27465526]   # km
    V = [-2.27456408, - 5.99222926, -0.77384874]         # km/sec
    R2, V2 = elementsToRV(*rvToElements(R, V))
    Rerr2 = R2 - R
    Verr2 = V2 - V

    print 'First Case'
    print '----------'
    print '{:s}:  [ {:14.8g}  {:14.8g}  {:14.8g} ]'.format('R', *Rerr1)
    print '{:s}:  [ {:14.8g}  {:14.8g}  {:14.8g} ]'.format('V', *Verr1)

    print '\nSecond Case'
    print '-----------'
    print '{:s}:  [ {:14.8g}  {:14.8g}  {:14.8g} ]'.format('R', *Rerr2)
    print '{:s}:  [ {:14.8g}  {:14.8g}  {:14.8g} ]'.format('V', *Verr2)

    # Second problem
    print '\n     Error Check for Second Problem'
    print '=========================================='

    # First case
    a = 9000.0      # km
    e = 0.050
    i = 40.0        # deg
    Omega = 45.0    # deg
    omega = 50.0    # deg
    nu = 20.0       # deg

    elements = [a, e, i, Omega, omega, nu]
    elements1 = rvToElements(*elementsToRV(*elements))
    errs1 = [el1 - el for el1, el in zip(elements1, elements)]

    # Second case
    a = 9400.0      # km
    e = 0.015
    i = 164.0       # deg
    Omega = 199.0   # deg
    omega = 310.0   # deg
    nu = 206.0      # deg

    elements = [a, e, i, Omega, omega, nu]
    elements2 = rvToElements(*elementsToRV(*elements))
    errs2 = [el2 - el for el2, el in zip(elements2, elements)]

    print '          First Case         Second Case'
    print '        --------------     --------------'
    for el, err1, err2 in zip(names, errs1, errs2):
        print '{:5s}   {:14.8g}      {:14.8g}'.format(el, err1, err2)


if __name__ == "__main__":
    __main__()
    print '\n\n'
    __test__()
