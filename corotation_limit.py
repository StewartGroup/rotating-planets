"""
Corotation limit (CoRoL) -- the maximum angular momentum a rotating terrestrial
planet of a given mass can have while remaining a single corotating body.

From Postema, Lock & Stewart (2026), "Angular momenta and pressure relationships
in rotating terrestrial planets", Table 2 (frozen magma-ocean thermal state).

    L_CoRoL / L_EM = k1 (M/M_earth)^k2 * <exp factor>
    L_EM = 3.5e34 kg m^2/s   (Postema's Earth-Moon normalisation)

Type 1 (Eq. 3) is the canonical simple power law, accurate for M >= 0.2 M_earth.
Type 4 (log, double-exponential) is the most accurate across low masses and is
used as a low-mass cross-check (it removes Type 1's ~30% high bias below 0.2 M).
"""
import numpy as np

L_EM_POSTEMA = 3.5e34     # kg m^2/s  (Postema normalisation)
J_EM_SURVEY  = 3.61e34    # kg m^2/s  (Timpe+23 / Meier+24 / our Raymond AM units)

# Table 2 coefficients: (k1, k2, k3, kind)
COEFFS = {
    1: (4.10531, 1.55849, 0.0,        'power'),    # Eq. 3
    2: (4.53621, 1.64019, -0.0883216, 'singleexp'),# k3 * M/M_earth        (log fit)
    3: (4.31904, 1.61064, -0.0457691, 'singleexp'),# k3 * M/M_earth        (linear fit)
    4: (3.78273, 1.64827,  0.221544,  'doubleexp'),# k3 * exp(-M/M_earth)  (log fit)
    5: (4.03092, 1.57542,  0.0501642, 'doubleexp'),# k3 * exp(-M/M_earth)  (linear fit)
}

def L_canup(M_earth, K=0.335, rho=3300.0, units='JEM_survey'):
    """
    Canup (2001) analytic spin-stability estimate L* (Postema+26 Eq. 8):
        L* = K M R^2 sqrt(GM/R^3) = K sqrt(G) (3/(4 pi rho))^(1/6) M^(5/3)
    the spin AM at which equatorial centrifugal force = gravity (rotational
    break-up of a SPHERICAL body).  Pure power law with exponent 5/3 = 1.667.

    K    : gyration constant I/MR^2 (0.4 uniform sphere, 0.335 Earth-like)
    rho  : body density (kg/m^3); ~3300 for uncompressed rock / small bodies.
           (Earth's mean rho=5510 with K=0.335 reproduces Postema's reference
            L*_Earth = 1.02e35 kg m^2/s = 2.91 L_EM.)
    """
    G, ME = 6.674e-11, 5.9722e24
    M = np.asarray(M_earth, dtype=float) * ME
    L_SI = K * np.sqrt(G) * (3.0/(4*np.pi*rho))**(1.0/6.0) * M**(5.0/3.0)
    if units == 'SI':
        return L_SI
    if units == 'LEM':
        return L_SI / L_EM_POSTEMA
    if units == 'JEM_survey':
        return L_SI / J_EM_SURVEY
    raise ValueError(units)


def L_corot(M_earth, type=1, units='JEM_survey'):
    """
    Maximum corotating angular momentum for a planet of mass M_earth (in M_earth).

    type   : 1..5  (Postema Table 2; 1 = canonical power law)
    units  : 'LEM'         -> L_CoRoL / L_EM   (dimensionless, Postema units)
             'JEM_survey'  -> in units of 3.61e34 kg m^2/s  (our Raymond AM units)
             'SI'          -> kg m^2 / s
    """
    M = np.asarray(M_earth, dtype=float)
    k1, k2, k3, kind = COEFFS[type]
    if kind == 'power':
        ratio = k1 * M**k2
    elif kind == 'singleexp':
        ratio = k1 * M**k2 * np.exp(k3 * M)
    elif kind == 'doubleexp':
        ratio = k1 * M**k2 * np.exp(k3 * np.exp(-M))
    if units == 'LEM':
        return ratio
    if units == 'SI':
        return ratio * L_EM_POSTEMA
    if units == 'JEM_survey':
        return ratio * L_EM_POSTEMA / J_EM_SURVEY   # convert to 3.61e34 units
    raise ValueError(units)


if __name__ == '__main__':
    for M in [0.05, 0.1, 0.2, 0.5, 1.0, 1.45]:
        print(f'M={M:>5} M_earth :  L_CoRoL(Type1)={L_corot(M):.3f}  '
              f'Type4={L_corot(M,4):.3f}  (J_EM=3.61e34 units)')
