#!/usr/bin/python

import numpy as np
from numpy import sin, cos, deg2rad, rad2deg
from numpy.linalg import norm

from kepler import julian_date, rvToElements, mu_sun, mu_earth

def dmsToDeg(dms):
    """Converts from deg:min:sec to decimal degree

    :dms: list or tuple with degree, minute, second
    :returns: angle in degrees

    """
    dmsDeg = sum([abs(e)/60.**i for i, e in enumerate(dms)])
    dmsDeg *= np.sign(dms[0])
    return dmsDeg


def hmsToDeg(hms):
    """Converts RA in hour:min:sec to decimal degree

    :hms: list or tuple with degree, minute, second
    :returns: angle in degrees

    """
    hmsToDecHour = sum([e/60.**i for i, e in enumerate(hms)])
    hmsDeg = 360*hmsToDecHour/24
    return hmsDeg


def __cosineVector__(ra, dec):
    """Generates matrix for transformation from perifocal
    coordinate system to geocentric-equatorial coordinate
    system.
    Note: All angles MUST be in DEGREES

    :i:     inclination
    :dec: longitude of ascending node
    :ra: argument of periapsis
    :returns: Transformation Matrix

    """
    ra = deg2rad(ra)
    dec = deg2rad(dec)

    return np.array([cos(dec)*cos(ra), cos(dec)*sin(ra), sin(dec)])


def __celestialTransform__(e, juliandate):
    """Transforms unit position vector from ECI to HCI

    :e: cosine position vector of celestial object
    :juliandate: Julian date for transform
    :returns: transformed cosine vector of celestial object

    """
    # Julian centuries Terestial Time since J2000 TT
    J2000 = 2451545.0
    Jcentury = 36525.0
    T = (juliandate-J2000)/Jcentury

    eps = deg2rad(23.43929111 + (-46.8150*T - 0.00059*T**2 + 0.001813*T**3)/3600)

    Rx = np.zeros((3,3))

    Rx[0, 0] = 1.0

    Rx[1, 1] =  cos(eps)
    Rx[1, 2] = -sin(eps)

    Rx[2, 1] =  sin(eps)
    Rx[2, 2] =  cos(eps)

    e_new = Rx.dot(e)

    return e_new

def __sectorToTriangle__(juliandates, rvec_a, rvec_b, ea, mu):
    """Determines the sector to triangle ratio from times and position vectors.

    :juliandates: Julian dates of observations
    :rvec_a: position vector for first observation
    :rvec_b: position vector for second observation
    :mu: mu for centric body (defaults to sun)
    :returns: orbital elements for orbiting body

    """
    tau = np.sqrt(mu)*np.diff(juliandates)[0]*24*3600

    ra = norm(rvec_a)
    rb = norm(rvec_b)
    r0 = norm(rvec_b - rvec_b.dot(ea)*ea)
    delta = 0.5*ra*r0

    m = tau**2/np.sqrt(2*(ra*rb+rvec_a.dot(rvec_b)))**3
    l = (ra+rb)/(2*np.sqrt(2*(ra*rb+rvec_a.dot(rvec_b)))) - 0.5

    g = lambda w: 2*np.arcsin(np.sqrt(w))
    W = lambda w: (2*g(w) - sin(2*g(w)))/sin(g(w))**3
    f = lambda eta: 1 - eta + (m/eta**2)*W(m/eta**2-l)

    eta0 = (12 + 10*np.sqrt(1+44*m/(9*(l+5./6.))))/22
    eta1 = eta0 + 0.1
    eta2 = eta0

    err = 1
    while abs(err) > 1e-10:
        eta3 = eta2 - f(eta2)*(eta2-eta1)/(f(eta2)-f(eta1))
        eta1, eta2 = eta2, eta3
        err = eta2 - eta1

    return eta3


def twoPositions(times, positions, mu=mu_earth):
    """Determines orbital elements from two position vectors using Gauss' method. Can
    either be used for Earth orbiting satellite, in which case positions are from center
    of Earth to surface observation location, or for heliocentric orbit, in which case
    positions are from Sun to Earth at particular times.

    :times: times for three observations in (YYYY, MM, DD, hh, mm, ss) format
    :positions: position vectors from inertial reference frame to observation points
    :mu: mu for centric body (defaults to sun)
    :returns: position and velocity vectors for observations

    """
    # Convert times from YYYY, MM, DD, hh:mm:ss to Julian date
    juliandates = np.array([julian_date(*time) for time in times])

    tau = np.sqrt(mu)*np.diff(juliandates)[0]*24*3600

    rvec_a, rvec_b = [np.array(r) for r in positions]
    ra = norm(rvec_a)
    rb = norm(rvec_b)

    ea = rvec_a/ra
    rvec_0 = rvec_b - rvec_b.dot(ea)*ea
    r0 = norm(rvec_0)
    e0 = rvec_0/r0

    # area of triangle from two position vectors
    delta = 0.5*ra*r0

    eta = __sectorToTriangle__(juliandates, rvec_a, rvec_b, ea, mu)

    W = np.cross(ea, e0)

    # argument of latitude
    u = np.arctan2(rvec_a[-1], -np.cross(rvec_a, W)[-1])

    # semi-latus rectum
    p = (2*delta*eta/tau)**2

    # eccentricity and true anomaly
    e_cos_nu = p/ra - 1
    e_sin_nu = ((p/ra - 1)*rvec_b.dot(ea)/rb - (p/rb - 1))/(r0/rb)
    e = np.sqrt(e_cos_nu**2 + e_sin_nu**2)
    nu = np.arctan2(e_sin_nu, e_cos_nu)

    # inclination
    i = np.arctan2(norm(W[:2]), W[2])

    # eccentric anomaly and mean anomaly
    E = np.arctan2(np.sqrt(1-e**2)*sin(nu), cos(nu)+e)
    M = E-e*sin(E)

    # right ascension of the ascending node
    Omega = np.arctan2(W[0], -W[1])

    # conversions from radians to degrees
    i = rad2deg(i) % 360
    Omega = rad2deg(Omega) % 360
    nu = rad2deg(nu) % 360

    u = rad2deg(u) % 360

    E = rad2deg(E) % 360
    M = rad2deg(M) % 360

    # argument of perigee
    omega = (u - nu) % 360

    # semi-major axis
    a = p/(1-e**2)

    return a, e, i, Omega, omega, nu


def threeAngles(times, positions, angles, heliocentric=False):
    """Determines orbital elements from three separate observations using Gauss' method.
    Can either be used for Earth orbiting satellite, in which case positions are from
    center of Earth to surface observation location, or for heliocentric orbit, in which
    case positions are from Sun to Earth at particular times.

    :times: times for three observations in (YYYY, MM, DD, hh, mm, ss) format
    :positions: distances from inertial reference frame to observation points
    :angles: right ascension (hms) and declination (dms) for the observations
    :heliocentric: flag to indicate heliocentric (True) or geocentric (False) orbit
    :returns: position vectors for observations

    """
    # Convert times from YYYY, MM, DD, hh:mm:ss to Julian date
    juliandates = np.array([julian_date(*time) for time in times])

    # Convert angles from HMS and DMS to decimal degree
    angles = [(hmsToDeg(angle[0]), dmsToDeg(angle[1])) for angle in angles]

    # Determine cosine position vector
    e = [__cosineVector__(angle[0], angle[1]) for angle in angles]

    if heliocentric:
        # Transform to celestial coordinates
        e = [__celestialTransform__(ei, jd) for ei, jd in zip(e, juliandates)]
        mu = mu_sun
    else:
        mu = mu_earth

    # Auxiliary unit vectors (Bucerius, 1950)
    d = [np.cross(e[(i+1)%3], e[(i+2)%3]) for i in xrange(3)]
    D0 = np.average([ei.dot(di) for ei, di in zip(e, d)])
    D = np.array([[di.dot(Rj) for Rj in positions] for di in d])
    # D = np.array([[di.dot(Rj) for di in d] for Rj in positions]).T()
    tau = np.array([(juliandates[(i+2)%3]-juliandates[(i+1)%3]) /
                  (juliandates[-1]-juliandates[0]) for i in xrange(3)])

    eta = np.ones(3)
    errors = np.ones(3)
    while max(abs(errors)) >1e-10:
        eta_old = eta.copy()
        n = eta[1]*tau/eta
        rho = abs(D.dot(n)/(n*D0))
        rhovec = [ri*ei for ri, ei in zip(rho, e)]
        rvec = [R + rho for R, rho in zip(positions, rhovec)]
        eta[0] = __sectorToTriangle__(juliandates[:2], rvec[0], rvec[1], e[0], mu)
        eta[1] = __sectorToTriangle__((juliandates[1:]), rvec[1], rvec[2], e[1], mu)
        eta[2] = __sectorToTriangle__(juliandates[[0, 2]], rvec[0], rvec[2], e[0], mu)
        errors = eta - eta_old
    return rvec


def main():
    times = ((2016, 11, 28, 23, 25, 0),
             (2016, 11, 29, 01, 05, 0),
             (2016, 11, 29, 02, 50, 0))

    RA = ((21, 31, 31),
          (18, 22, 25),
          (21, 24, 21))

    dec = ((+44, 07, 40),
           (+ 8, 22, 25),
           (-24, 36, 11))

    Re = 6378.137  # km
    lam = -85.483127
    phi = 32.605763
    H = 200.601  # m
    f = 1/298.256421867

    lam = np.deg2rad(lam)
    phi = np.deg2rad(phi)
    xc = (Re/np.sqrt(1-(2*f-f**2)*sin(phi)**2) + H/1000)*cos(phi)
    zc = (Re*(1-2*f-f**2)/np.sqrt(1-(2*f-f**2)*sin(phi)**2) + H//1000)*sin(phi)

    J2000 = julian_date(2000, 1, 1, 12, 0, 0)

    lams = [280.4606 + 360.9856473*(julian_date(*time)-J2000) for time in times]
    positions = [(xc*cos(lam), xc*sin(lam), zc) for lam in lams]
    print positions

    angles = zip(RA, dec)

    rvec = threeAngles(times, positions, angles)
    print rvec

    times = times[0], times[-1]
    rvec = rvec[0], rvec[-1]
    return twoPositions(times, rvec, mu_earth)


if __name__ == "__main__":

    a, e, i, Omega, omega, nu = main()

    # times = ((1999, 04, 02, 00, 30, 00.0),
    #          (1999, 04, 02, 03, 00, 00.0))

    # positions = ((11959.978, -16289.478, -5963.827),
    #              (39863.390, -13730.547, -4862.350))

    # a, e, i, Omega, omega, nu = twoPositions(times, positions, mu=mu_earth)

    print 'a: ', a
    print 'e: ', e
    print 'i: ', i
    print 'Omega: ', Omega
    print 'omega: ', omega
    print 'nu: ', nu
