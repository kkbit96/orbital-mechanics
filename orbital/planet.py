import numpy as np

au = 149597870.691      # km

class Planet(object):

    """Class representing a planetary body, with orbital elements position and velocity vectors."""

    def __init__(self, name, date_time):
        """Initializes planetary object, importing ephemeride data and setting initial orbital
        elements and position and velocity vectors.

        :name: name of planet to be modelled
        :date_time: tuple or list with date and time representing initial or current time
                    format: (YEAR, MONTH, DAY, HOUR, MINUTE)

        """

        self.name = name.capitalize()
        self.juliandate = self.julian_date(*date_time)
        self.__import_ephemerides__(self.name)
        self.__compute_vectors__()

    @staticmethod
    def julian_date(year, month, day, hour, minute, seconds):
        """Computes Julian date for specified date vector.

        :date_time: list or tuple with year, month, day, hour, minute
        :returns: Julian date number.

        """
        j0 = 367*year - np.floor(7*(year + np.floor((month+9)/12))/4) + np.floor(275*month/9) + day + 1721013.5
        jd = j0 + hour/24. + (minute+seconds/60.)/1440.

        return jd

    def __import_ephemerides__(self, planet):
        """Import planetary ephemeride tables from package ephemerides file

        :returns: dict of ephemeride information

        """

        import os

        directory = os.path.split(__file__)[0]
        filename = os.path.join(directory, 'ephemerides')

        ephemerides = {}
        with open(filename, 'r') as ephemeride_file:
            for line in ephemeride_file:
                row = line.split()
                if len(row) == 0:
                    pass
                elif len(row) == 1:
                    name = row[0].capitalize()
                else:
                    if planet == name:
                        ephemerides[row[0]] = [float(num) for num in row[1:]]

        self.ephemerides = ephemerides

    def __compute_vectors__(self):

        from numpy.polynomial.polynomial import polyval

        from kepler import elementsToRV, mu_sun

        T = (self.juliandate - 2451545.0)/36525.0

        self.L = polyval(T, self.ephemerides["L"]) % 360
        self.i = polyval(T, self.ephemerides["i"]) % 360
        self.Omega = polyval(T, self.ephemerides["Omega"]) % 360
        self.pi = polyval(T, self.ephemerides["pi"]) % 360
        self.a = polyval(T, self.ephemerides["a"]) * au
        self.e = polyval(T, self.ephemerides["e"])

        self.omega = (self.pi - self.Omega) % 360
        if self.name == 'PLUTO':
            self.M = (self.L - self.pi - 0.01262724 * T**2) % 360
        else:
            self.M = (self.L - self.pi) % 360

        M = np.deg2rad(self.M)

        E = M + self.e * np.sin(M)

        err = 1
        while abs(err) > 1e-10:
            E_old = E
            err = (E - self.e*np.sin(E) - M)/(1-self.e*np.cos(E))
            E = E - err

        nu = 2*np.arctan(np.sqrt((1+self.e)/(1-self.e))*np.tan(E/2))

        self.E = np.rad2deg(E) % 360
        self.nu = np.rad2deg(nu) % 360

        self.R, self.V = elementsToRV(self.a, self.e, self.i, self.Omega,
                                      self.omega, self.nu, mu_sun)

    def update_orbit(self, dt):
        """Updates orbit for one time step of dt

        :dt: time step in fraction of day

        """

        self.juliandate += dt
        self.__compute_vectors__()
