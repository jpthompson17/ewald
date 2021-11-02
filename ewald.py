from math import inf, isinf, pi, sqrt, exp, erfc
from itertools import product


def madelung(screening_length=inf, g_ewald=7.1, kmax=12, approx=False):
    """Compute the Madelung constant for a 1x1x1 Yukawa lattice with a 
    uniform neutralizing background.

 
    Parameters
    ----------
    screening_length : float, default inf
        The dimensionless screening length. The default corresponds
        to the bare Coulomb potential.
    g_ewald : float, default 7.1
        Ewald splitting parameter.
    kmax : int, default 12
        Range of the reciprocal space sum.
    approx : bool, default False
        Whether to use the closed-form approximation given by [??].

    Notes
    -----
    See eqn (2.19) of Salin and Caillol, J. Chem. Phys. 113, 10459 (2000).
    (alpha = 1/screening_length, beta = g_ewald)

    """

    if isinf(screening_length):
        return -2.837297479
    if screening_length == 0:
        return 0

    a = 1.0 / screening_length  # screening wavenumber
    a2 = a * a

    # If approx is True, return the closed-form approximation
    if approx:
        r = 0.911544
        r2 = r * r
        return (4*pi/a2 * (a*r + 1 + a2*r2/3) - 1/r)*exp(-a*r) - 4*pi/a2

    b = 2 * g_ewald 
    b2 = b * b
    kmax2 = kmax * kmax
    pi2 = pi * pi

    s = 0  # running total

    # Sum over the (+,+,+) octant, then multiply the result by 8.
    for k in list(product(range(kmax + 1), repeat=3))[1:]:
        k2 = k[0]*k[0] + k[1]*k[1] + k[2]*k[2]
        if k2 <= kmax2:
            q2 = 4 * pi2 * k2 # squared wavenumber
            t = exp(-(q2 + a2) / b2) / (q2 + a2)
            for i in range(3):
                # Correct for overcounting when k is on an axial plane. 
                if k[i] == 0:
                    t /= 2
            s += t
    # This is where we multiply by 8.
    # (We also need a 4*pi because we are using Gaussian units.)
    s *= 32 * pi 

    # See last line of eqn (2.19) of Salin and Caillol.
    # (Note the sign error in the final term.)
    s -= b * exp(-a2 / b2) / sqrt(pi)
    s += a * erfc(a / b)
    if a > 0.001:
        s += 4*pi * (exp(-a2 / b2) - 1) / a2
    else:  # If a is small, expand about a = 0.
        s += -4*pi/b2 + 2*pi*a2/(b2*b2) - 2*pi*a2*a2/(3*b2*b2*b2)

    return s

