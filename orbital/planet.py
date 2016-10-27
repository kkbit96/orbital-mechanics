import numpy as np

au = 149597870.691      # km

class Planet(object):

    """Class representing a planetary body, with orbital elements position and velocity vectors."""
    from datetime import datetime

    def __init__(self, name, date_time):
        """Initializes planetary object, importing ephemeride data and setting initial orbital
        elements and position and velocity vectors.

        :name: name of planet to be modelled
        :date_time: tuple or list with date and time representing initial or current time
                    format: (YEAR, MONTH, DAY, HOUR, MINUTE)

        """

        self.name = name.upper()
        self.juliandate = self.julian_date(*date_time)
        self.__import_ephemerides__(self.name)
        self.__compute_vectors__()

    @staticmethod
    def julian_date(year, month, day, hour, minute):
        """Computes Julian date for specified date vector.

        :date_time: list or tuple with year, month, day, hour, minute
        :returns: Julian date number.

        """
        # j0 = 367*year - int(7*(year + int((month+9)/12))/4) + int(275*month/9) + day + 1721013.5
        j0 = 367*year - np.floor(7*(year + np.floor((month+9)/12))/4) + np.floor(275*month/9) + day + 1721013.5
        jd = j0 + hour/24. + (minute+5/60.)/1440.

        return jd


    def __import_ephemerides__(self, planet, filename='ephemerides'):
        """Import planetary ephemeride tables from specified file (defaults to ephemerides.txt).

        :filename: file name containing ephemeride information
        :returns: dict of ephemeride information

        """
        ephemerides = {}
        with open(filename, 'r') as ephemeride_file:
            for line in ephemeride_file:
                row = line.split()
                if len(row) == 0:
                    pass
                elif len(row) == 1:
                    name = row[0]
                else:
                    if planet == name:
                        ephemerides[row[0]] = [float(num) for num in row[1:]]

        self.ephemerides = ephemerides

    def __compute_vectors__(self):

        from numpy.polynomial.polynomial import polyval

        from kepler import elementsToRV, mu_sun

        T = (self.juliandate - 2451545.0)/36525.0

        L = polyval(T, self.ephemerides["L"]) % 360
        i = polyval(T, self.ephemerides["i"])
        Omega = polyval(T, self.ephemerides["Omega"]) % 360
        pi = polyval(T, self.ephemerides["pi"])
        a = polyval(T, self.ephemerides["a"]) * au
        e = polyval(T, self.ephemerides["e"])

        omega = pi - Omega
        if self.name == 'PLUTO':
            M = (L - pi - 0.01262724 * T**2)
        else:
            M = (L - pi)

        M = np.deg2rad(M % 360)

        E = M + e * np.sin(M)

        err = 1
        while abs(err) > 1e-10:
            E_old = E
            # E = (M + e*np.sin(E))
            # err = E - E_old
            err = (E - e*np.sin(E) - M)/(1-e*np.cos(E))
            E = E - err

        nu = 2*np.arctan(np.sqrt((1+e)/(1-e))*np.tan(E/2))

        nu = np.rad2deg(nu)

        self.R, self.V = elementsToRV(a, e, i, Omega, omega, nu, mu_sun)

    def update_orbit(self, dt):
        """Updates orbit for one time step of dt

        :dt: time step in fraction of day

        """

        self.juliandate += dt
        self.__compute_vectors__()
