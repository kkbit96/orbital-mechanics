from planet import Planet
import numpy as np

class Moon(Planet):

    """Class representing a planetary body, with orbital elements position and velocity vectors."""

    def __import_ephemerides__(self, name):
        """Import planetary ephemeride tables from specified file (defaults to ephemerides.txt).

        :filename: file name containing ephemeride information
        :returns: dict of ephemeride information

        """
        import os

        directory = os.path.split(__file__)[0]

        ephemerides = {
            'Lprime': [218.3164477, 481267.88123421, -0.0015786, 1/538841., -1/6.5194e7],
            'D': [297.8501921, 445267.1114034, -0.0018819, 1/545868., -1/1.13065e8],
            'M': [357.5291092, 35999.0502909, -0.0001536, 1./2.449e7],
            'Mprime': [134.9633964, 477198.8675055, 0.0087414, 1/69699., -1/1.4712e7],
            'F': [93.2720950, 483200.0175233, -0.0036539, -1/3.526e6, 1/8.6331e8],
            'Omega': [125.04452, -1934.136261, 0.0020708, 1/4.5e5],
            'L': [280.46646, 36000.76983, 0.0003032],
            'A1': [119.75, 131.849],
            'A2': [53.09, 479264.290],
            'A3': [313.45, 481266.484],
            'E': [1.00, -0.002516, -0.0000074]
        }

        # Import lunar latitude table
        moon_files = 'moon_nutation_obliquity', 'moon_longitude_distance', 'moon_latitude'
        for moon_file in moon_files:

            filename = os.path.join(directory, moon_file)
            with open(filename, 'r') as ephemeride_file:
                table = []

                for line in ephemeride_file:
                    params = [col.split() for col in line.split('|')]
                    table_row = [[int(num) for num in params[0]]]

                    if len(params[1]) > 0:
                        table_row.append([float(num) for num in params[1]])
                    else:
                        table_row.append([0])

                    if len(params) == 3:
                        if len(params[2]) > 0:
                            table_row.append([float(num) for num in params[2]])
                        else:
                            table_row.append([0])

                    table.append(table_row)

                ephemerides[moon_file.split('_')[1]] = table

        self.ephemerides = ephemerides

    @staticmethod
    def __table_calcs__(arguments, table, T, E=1):

        from numpy.polynomial.polynomial import polyval

        val = [0 for _ in xrange(len(table[0])-1)]

        for row in table:
            factors = row[0]
            arg = np.deg2rad(sum([f*a for f, a in zip(factors, arguments)]))

            if abs(factors[1]) == 1:
                mult = E
            elif abs(factors[1]) == 2:
                mult = E**2
            else:
                mult = 1

            for i, poly, trig_fun in zip(range(2), row[1:], (np.sin, np.cos)):
                coeff = mult*polyval(T, poly)
                val[i] += coeff*trig_fun(arg)

        return val

    def __compute_vectors__(self):

        from numpy.polynomial.polynomial import polyval
        from numpy import sin, cos, deg2rad

        T = (self.juliandate - 2451545.0)/36525.0

        self.Lprime = polyval(T, self.ephemerides['Lprime']) % 360
        self.D = polyval(T, self.ephemerides['D']) % 360
        self.M = polyval(T, self.ephemerides['M']) % 360
        self.Mprime = polyval(T, self.ephemerides['Mprime']) % 360
        self.F = polyval(T, self.ephemerides['F']) % 360
        self.Omega = polyval(T, self.ephemerides['Omega']) % 360
        # self.L = polyval(T, self.ephemerides['L'])
        self.A1 = polyval(T, self.ephemerides['A1']) % 360
        self.A2 = polyval(T, self.ephemerides['A2']) % 360
        self.A3 = polyval(T, self.ephemerides['A3']) % 360
        self.E = polyval(T, self.ephemerides['E'])


        args = [self.D, self.M, self.Mprime, self.F, self.Omega]

        # Computation of delta psi for lunar nutation
        self.delta_psi, self.delta_eps = \
            self.__table_calcs__(args, self.ephemerides['nutation'], T, self.E)

        self.delta_psi /= 3600e4
        self.delta_eps /= 3600e4

        # Computation of periodic permutations of longitude and distance
        self.sigma_l, self.sigma_r = \
            self.__table_calcs__(args, self.ephemerides['longitude'], T, self.E)

        # Computation of periodic permutations of latitude
        self.sigma_b, = \
            self.__table_calcs__(args, self.ephemerides['latitude'], T, self.E)

        # Conversion for angles necessary for additional terms
        Lprime = np.deg2rad(self.Lprime)
        Mprime = np.deg2rad(self.Mprime)
        F = np.deg2rad(self.F)
        A1 = np.deg2rad(self.A1)
        A2 = np.deg2rad(self.A2)
        A3 = np.deg2rad(self.A3)

        self.sigma_l += 3958*sin(A1) + 1962*sin(Lprime-F) + 318*sin(A2)
        self.sigma_l *= 1e-6

        self.sigma_b += -2.235*sin(Lprime) + 382*sin(A3) + 175*sin(A1-F) \
            + 175*sin(A1+F) + 127*sin(Lprime-Mprime) - 115*sin(Lprime+Mprime)
        self.sigma_b *= 1e-6

        self.sigma_r *= 1e-3

        self.lam = (self.Lprime + self.sigma_l + self.delta_psi) % 360
        self.beta = self.sigma_b % 360
        self.delta = 385000.56 + self.sigma_r

        cos_beta = cos(deg2rad(self.beta))
        sin_beta = sin(deg2rad(self.beta))

        cos_lambda = cos(deg2rad(self.lam))
        sin_lambda = sin(deg2rad(self.lam))

        self.R = self.delta*np.array([cos_beta*cos_lambda, cos_beta*sin_lambda, sin_beta])

    def update_orbit(self, dt):
        """Updates orbit for one time step of dt

        :dt: time step in fraction of day

        """

        self.juliandate += dt
        self.__compute_vectors__()
