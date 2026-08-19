"""
Modified specific impact energy Q_S of Leinhardt & Stewart (2012), as used by
Lock & Stewart (2017) Fig 12A (the S_upper(Q_S) hot-CoRoL calibration) and by
Quintana et al. (2016).  This is the SAME quantity Quintana_data_structure.calc_QS
computes; the bare reduced-mass Q_R = 0.5 mu v^2/M_tot is NOT it.

  Q_R   = mu v_imp^2 / (2 M_tot)                    reduced-mass specific energy
  B     = (R_t + R_p) b                              impact-parameter lever (b = sin theta)
  l     = 2 R_p                if B + R_p <= R_t      (projectile fully engulfed)
        = R_t + R_p - B        otherwise             (interacting length)
  alpha = 1                    if B + R_p <= R_t
        = (3 R_p l^2 - l^3)/(4 R_p^3)  otherwise     (interacting-mass fraction)
  mu_a  = alpha M_p M_t /(alpha M_p + M_t)
  Q'_R  = (mu_a/mu) Q_R                              interacting-mass corrected
  Q_S   = Q'_R (1 + M_p/M_t)(1 - b)                  modified, grazing-aware

alpha depends only on the mass ratio (-> radius ratio at equal density) and b, so
the absolute density cancels; pass radii if you have them, else they are built from
`rho` (equal-density spheres).  Masses in kg, v_imp in m/s, b dimensionless.
The larger of (M_t, M_p) is taken as the target, per the S&L2012 convention.
"""
import numpy as np


def QS_modified(M_t, M_p, v_imp, b, R_t=None, R_p=None, rho=3000.0):
    M_t, M_p = np.asarray(M_t, float), np.asarray(M_p, float)
    v_imp, b = np.asarray(v_imp, float), np.asarray(b, float)
    # target = larger body
    Mt = np.maximum(M_t, M_p)
    Mp = np.minimum(M_t, M_p)
    if R_t is None or R_p is None:
        Rt = (3*Mt/(4*np.pi*rho))**(1/3)
        Rp = (3*Mp/(4*np.pi*rho))**(1/3)
    else:
        # keep radii paired with the (possibly swapped) target/projectile masses
        R_t, R_p = np.asarray(R_t, float), np.asarray(R_p, float)
        swap = M_p > M_t
        Rt = np.where(swap, R_p, R_t)
        Rp = np.where(swap, R_t, R_p)

    mu   = Mt*Mp/(Mt+Mp)
    Mtot = Mt+Mp
    QR   = 0.5*mu*v_imp**2/Mtot

    B = (Rt+Rp)*b
    engulf = (B+Rp) <= Rt
    l = np.where(engulf, 2*Rp, Rt+Rp-B)
    alpha = np.where(engulf, 1.0, (3*Rp*l**2 - l**3)/(4*Rp**3))
    alpha = np.clip(alpha, 0.0, 1.0)
    mu_a = alpha*Mp*Mt/(alpha*Mp+Mt)
    QRp  = (mu_a/mu)*QR
    return QRp*(1+Mp/Mt)*(1-b)
