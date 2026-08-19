"""
Scaling law for the angular momentum bound to the largest remnant (J_bound)
in the Moon-forming giant-impact surveys of Timpe et al. (2023, ApJ 959:38)
and Meier et al. (2024, ApJ 978:11).

Neither paper parameterised J_bound; this provides a closed-form fit.

----------------------------------------------------------------------------
PHYSICAL BASIS
----------------------------------------------------------------------------
Angular momentum is overwhelmingly *retained* by the bound remnant, so the law
is built on the signed angular-momentum budget (z-axis = orbital AM direction):

    J0  =  J_orb  +  J_spin,targ  +  J_spin,proj          [Meier+24 Eq. 5]

Each component is retained with its own efficiency.  Ejecta carry away orbital
AM, and ejecta loss grows for small (low gamma), fast (high v_inf), grazing
(high b_inf) impactors; spins are retained almost fully.  Hence:

    J_bound  =  a(gamma, v_inf, b_inf) * J_orb
                + b * J_spin,targ  +  c * J_spin,proj

    a(.) = exp( -k * (v_inf)^p * (1 + b_inf)^q / (1 + gamma)^r )

a is the orbital-AM retention efficiency (0 < a <= 1).  Report |J_bound|.

----------------------------------------------------------------------------
INPUTS (the survey's dimensionless impact parameters)
----------------------------------------------------------------------------
    gamma  = M_proj / M_targ                    mass ratio        (col 'gamma')
    v_inf  = v_inf / v_esc                       asymptotic speed  (col 'v_inf')
    b_inf  = b_inf / R_grav                      asymptotic imp. p (col 'b_inf')
    J_orb, J_spin,targ, J_spin,proj              signed AM, in J_EM

Signed AM components from the survey columns (J_EM = 3.61e34 kg m^2/s):
    s_t = {U:+1, D:-1, N:0}[rot_config[0]]       (target spin sense)
    s_p = {U:+1, D:-1, N:0}[rot_config[1]]       (projectile spin sense)
    J_spin,targ = s_t * J_rot_targ / J_EM
    J_spin,proj = s_p * J_rot_proj / J_EM
    J0          = J_tot_init_sign * J_tot_init / J_EM
    J_orb       = J0 - J_spin,targ - J_spin,proj

----------------------------------------------------------------------------
FIT QUALITY  (5-fold cross-validated, N = 6473 events with defined J_bound)
----------------------------------------------------------------------------
    R^2  = 0.995        RMSE = 0.109 J_EM
    median relative error 2.3 %, 90th-pct 7.9 %
    cf. naive J_bound = J0:  RMSE = 0.211 J_EM  (this law halves the scatter)
"""
import numpy as np

# fitted coefficients (full-sample, saturating retention form)
K, P, Q, R = 0.1148, 0.777, 3.493, 5.110     # orbital-loss term
B, C       = 0.975, 0.920                      # target / projectile spin retention

# coefficients refit on the non-rotating (NN / Timpe+23) subset alone
# (ORIGINAL Earth-mass-only fit, in ASYMPTOTIC variables v_inf, b_inf)
K_NR, P_NR, Q_NR, R_NR = 0.1588, 1.139, 3.104, 4.997

# UPDATED non-rotating retention, fit on Timpe-Meier(NN) + Postema(2025)
# CLEAN-ACCRETION subset (xi=(M_bnd-M_t)/M_i >= 0.8) over a BROAD mass range
# (M_t = 0.01-1.3 M_earth), in CONTACT-frame variables.  Graze-and-merge /
# partial-accretion cases (0.2 < xi < 0.8) are NOT described by this smooth law
# (they lose much more AM, ~0.3 lower eta) and are handled by the ejecta law below
# (J_bound_grazemerge).
# eta = exp[-k (v_imp/v_esc)^p (sin theta)^q / (1+gamma)^r]   (loss at grazing)
K_NRB, P_NRB, Q_NRB, R_NRB = 1.0064, 3.248, 7.537, 3.689

# EJECTA-BASED graze-and-merge / accretionary retention (S. Stewart 2026; see
# ANALYSIS_ejecta_graze_merge_AM.md).  Models the AM CARRIED OFF BY EJECTA:
#   1 - eta = A (1+b)^m f_ej^s,   f_ej = M_ej/M_tot = 1 - M_lr/M_tot
# so   eta = 1 - A (1+b)^m f_ej^s   (clipped to [0,1]).
# (The specific-AM enhancement Lambda=(1-eta)/f_ej = A(1+b)^m f_ej^(s-1) is a
# separate diagnostic; do not confuse the coefficient A(1+b)^m with Lambda since s!=1.)
# Fit on the ACCRETIONARY (xi>0.2) Timpe-Meier(NN)+Postema set INCLUDING graze-
# and-merge (N=486; R2(log-loss)=0.76).  Erosive/disruptive (xi<0.2) and hit-and-
# run are a SEPARATE regime, NOT covered here (pinned for later work).
A_GM, M_GM, S_GM = 4.362941299820175, 1.274173427265838, 1.101167386820984
# geometry -> mass-loss estimator (for N-body sets that do not resolve ejecta):
#   f_ej = exp[ c0 ln(v/vesc) + c1 ln(1+b) + c2 ln(1+gamma) + c3 ]  (R2(log)=0.59)
FEJ_C = np.array([6.452788556852796, 3.1110087434610505,
                  -2.628257547514096, -5.757718109631945])

J_EM = 3.61e34   # kg m^2 / s

# ---------------------------------------------------------------------------
# RANGE OF VALIDITY
# ---------------------------------------------------------------------------
# The orbital-retention factor a(gamma, v_inf, b_inf) is only *directly*
# constrained by the non-rotating (NN) runs -- in rotating runs with a small
# impactor the orbital AM is a minor part of the budget, so those points give
# the fit almost no leverage on a.  The NN coverage is dense for gamma >= 0.3
# (320 runs), thin at gamma = 0.1 (10 runs), and below gamma ~ 0.05 the few
# runs are erosive (M_ej >~ M_imp) AND degenerate (small gamma only sampled at
# high v_inf).  There the retention-fraction picture breaks down (ejecta
# excavate target material; bound AM is not simply a fraction of the impact's
# orbital AM) and the law is an unconstrained extrapolation.
# We therefore restrict the law to the mass-ratio / velocity box that the
# survey actually covers:
GAMMA_MIN, GAMMA_MAX = 0.1, 1.0      # robustly constrained for gamma >= 0.3
V_INF_MIN, V_INF_MAX = 0.1, 3.0      # gridded asymptotic-speed range (v_esc)
B_INF_MAX            = 1.25          # gridded asymptotic impact-parameter range

def in_validity_range(gamma, v_inf, b_inf):
    """Boolean (array) True where the inputs lie inside the calibrated box."""
    g, v, b = np.asarray(gamma), np.asarray(v_inf), np.asarray(b_inf)
    return ((g >= GAMMA_MIN) & (g <= GAMMA_MAX) &
            (v >= V_INF_MIN) & (v <= V_INF_MAX) & (b <= B_INF_MAX))


def orbital_retention(gamma, v_inf, b_inf, enforce_range=True):
    """
    Orbital angular-momentum retention efficiency a in (0, 1].

    enforce_range : if True (default), returns NaN outside the calibrated box
    (see GAMMA_MIN etc.) instead of silently extrapolating into the small/fast
    erosive regime where the law is not constrained.
    """
    L = K * np.asarray(v_inf)**P * (1.0 + np.asarray(b_inf))**Q \
        / (1.0 + np.asarray(gamma))**R
    a = np.exp(-L)
    if enforce_range:
        a = np.where(in_validity_range(gamma, v_inf, b_inf), a, np.nan)
    return a


def J_bound(gamma, v_inf, b_inf, J_orb, J_spin_targ=0.0, J_spin_proj=0.0,
            enforce_range=True):
    """
    Predicted angular momentum bound to the largest remnant, in units of J_EM.

    Parameters (scalars or arrays)
        gamma        : mass ratio M_proj / M_targ
        v_inf        : asymptotic relative speed in units of v_esc
        b_inf        : asymptotic impact parameter in units of R_grav
        J_orb        : signed orbital AM (J_EM units)
        J_spin_targ  : signed target spin AM (J_EM units), default 0 (non-rotating)
        J_spin_proj  : signed projectile spin AM (J_EM units), default 0
        enforce_range: NaN outside the calibrated box (default True); see
                       in_validity_range / GAMMA_MIN.  Do NOT apply this law to
                       gamma < 0.1 (e.g. planetesimal impactors) -- the orbital
                       AM there is erosive and the law is unconstrained.

    Returns
        signed bound AM (J_EM units).  Use abs() to compare with the survey's
        magnitude column 'J_bound'.
    """
    a = orbital_retention(gamma, v_inf, b_inf, enforce_range=enforce_range)
    return a * np.asarray(J_orb) + B * np.asarray(J_spin_targ) + C * np.asarray(J_spin_proj)


def J_bound_nonrotating(gamma, v_inf, b_inf, J0, enforce_range=True):
    """
    Single-efficiency law for NON-ROTATING bodies (Timpe+23, or NN config):
        J_bound = eta(gamma, v_inf, b_inf) * J0,   eta = exp(-L_NR)
    Coefficients refit on the non-rotating subset (R^2=0.992, RMSE=0.072 J_EM).
    J0 in J_EM units; equals J_orb when there is no spin.
    Valid only inside the calibrated box (enforce_range=True -> NaN outside).
    """
    L = K_NR * np.asarray(v_inf)**P_NR * (1.0 + np.asarray(b_inf))**Q_NR \
        / (1.0 + np.asarray(gamma))**R_NR
    eta = np.exp(-L)
    if enforce_range:
        eta = np.where(in_validity_range(gamma, v_inf, b_inf), eta, np.nan)
    return eta * np.asarray(J0)


def J_bound_nonrotating_broad(gamma, v_imp_over_vesc, sin_theta, J0):
    """
    UPDATED non-rotating J_bound law over a broad mass range (0.01-1.3 M_earth),
    fit on the Timpe-Meier(NN) + Postema(2025) CLEAN-ACCRETION subset (xi>=0.8):

        J_bound = eta * J0,
        eta = exp[ -k (v_imp/v_esc)^p (sin theta)^q / (1+gamma)^r ]

    Retention is ~0.98 up to theta~60 deg and drops to ~0.73 only at very grazing
    angles.  CONTACT-frame inputs (what SPH surveys report directly):
        gamma            = M_i / M_t            (impactor/target mass ratio)
        v_imp_over_vesc  = impact speed / mutual escape speed  (>= 1)
        sin_theta        = sin(impact angle) = impact parameter b  (0 head-on .. 1 grazing)
        J0               = initial orbital angular momentum (any consistent units)
    Returns J_bound in the same units as J0.  Approximately scale-free.

    NOT VALID for graze-and-merge / partial accretion (accretion efficiency
    xi=(M_bnd-M_t)/M_i < 0.8): those lose much more AM (eta ~0.3 lower, scattered).
    Use J_bound_grazemerge (ejecta law) for them, or J_bound_accretionary to switch
    automatically by xi/regime.
    """
    g = np.asarray(gamma); v = np.asarray(v_imp_over_vesc); s = np.asarray(sin_theta)
    L = K_NRB * v**P_NRB * s**Q_NRB / (1.0 + g)**R_NRB
    return np.exp(-L) * np.asarray(J0)


def f_ej_predict(v_imp_over_vesc, sin_theta, gamma):
    """
    LAST-RESORT geometric estimate of the mass-lost (ejecta) fraction
    f_ej = M_ej/M_tot = 1 - M_lr/M_tot, for the rare case where the largest-remnant
    mass is unavailable:

        f_ej = exp[ c0 ln(v/vesc) + c1 ln(1+b) + c2 ln(1+gamma) + c3 ]   b=sin_theta

    Fit on the accretionary (xi>0.2) SPH set, R2(log)=0.59.  In practice you should
    almost never need this: every collision record reports M_lr and M_t+M_p, so the
    MEASURED f_ej = 1 - M_lr/M_tot is available and strongly preferred.  Clipped to
    the physical range [0,1].
    """
    v = np.asarray(v_imp_over_vesc); b = np.asarray(sin_theta); g = np.asarray(gamma)
    f = np.exp(FEJ_C[0]*np.log(v) + FEJ_C[1]*np.log(1.0+b)
               + FEJ_C[2]*np.log(1.0+g) + FEJ_C[3])
    return np.clip(f, 0.0, 1.0)          # mass-lost fraction is physical in [0,1]


def eta_grazemerge(sin_theta, f_ej):
    """
    Ejecta-based AM-retention efficiency for accretionary / graze-and-merge impacts:
        eta = 1 - A (1+b)^m f_ej^s,   clipped to [0,1].
    b = sin_theta;  f_ej = mass-lost fraction M_ej/M_tot (= 1 - M_lr/M_tot), which
    is itself clipped to the physical range [0,1].
    """
    b = np.asarray(sin_theta); f = np.clip(np.asarray(f_ej), 0.0, 1.0)
    return np.clip(1.0 - A_GM * (1.0 + b)**M_GM * f**S_GM, 0.0, 1.0)


def J_bound_grazemerge(gamma, v_imp_over_vesc, sin_theta, J0, f_ej=None):
    """
    Bound AM for ACCRETIONARY impacts INCLUDING graze-and-merge, via the ejecta law:

        J_bound = eta * J0,   eta = 1 - A (1+b)^m f_ej^s

    Unlike J_bound_nonrotating_broad (pure geometry, clean accretion only), this
    folds in the graze-and-merge / partial-accretion events by keying on the mass
    actually lost.  Inputs:
        gamma, v_imp_over_vesc, sin_theta : contact-frame impact parameters
        J0    : initial orbital AM (any consistent units)
        f_ej  : measured mass-lost fraction M_ej/M_tot (= 1 - M_lr/M_tot).  If None,
                estimated from geometry via f_ej_predict() -- use for N-body sets
                with unresolved ejecta; pass the measured value for SPH.
    Returns J_bound in the same units as J0.  Covers xi>0.2 (accretion + graze-and-
    merge); NOT valid for erosive/disruptive (xi<0.2) or hit-and-run impacts.
    """
    if f_ej is None:
        f_ej = f_ej_predict(v_imp_over_vesc, sin_theta, gamma)
    return eta_grazemerge(sin_theta, f_ej) * np.asarray(J0)


def J_bound_accretionary(gamma, v_imp_over_vesc, sin_theta, J0,
                         f_ej=None, xi=None, use_clean=False):
    """
    Bound AM for accretionary impacts, INCLUDING graze-and-merge.

    DEFAULT is the ejecta law (J_bound_grazemerge), which spans the whole
    accretionary range and is more accurate than the pure-geometry clean law
    (RMS 0.10 vs 0.16 in eta).  The switch is deliberately NOT keyed on xi: graze-
    and-merge events span all xi (the high-xi grazing mergers of e.g. Clement+19
    have xi=0.89-1.0), so an xi threshold would misroute exactly the population the
    ejecta law is for.  Instead we exploit the fact that the mass-lost fraction

        f_ej = 1 - M_lr / M_tot          (M_lr = largest remnant, M_tot = M_t+M_i)

    is essentially ALWAYS available -- pass it, and the ejecta law is used for every
    (non-erosive) event.  If f_ej is omitted it is estimated from geometry
    (f_ej_predict), a last resort you should rarely need.

        f_ej      : measured mass-lost fraction M_ej/M_tot (= 1 - M_lr/M_tot).
        xi        : if given, erosive / hit-and-run events (xi <= 0.2) return NaN
                    (that regime is not covered by either law).
        use_clean : if True, use the pure-geometry clean-accretion law
                    (J_bound_nonrotating_broad, the published Eq.~eta) instead;
                    it ignores f_ej.  Kept for reference / reproducing that fit.
    """
    if use_clean:
        out = J_bound_nonrotating_broad(gamma, v_imp_over_vesc, sin_theta, J0)
    else:
        out = J_bound_grazemerge(gamma, v_imp_over_vesc, sin_theta, J0, f_ej=f_ej)
    if xi is not None:                               # erosive / hit-and-run: not covered
        out = np.where(np.asarray(xi, float) <= 0.2, np.nan, np.asarray(out, float))
    return out


def J_bound_from_dataframe(df, enforce_range=True):
    """Convenience wrapper: predict J_bound (J_EM units) for the survey CSV.
    With enforce_range=True, out-of-validity rows (gamma<0.1 etc.) return NaN."""
    sgn = {'U': 1.0, 'D': -1.0, 'N': 0.0}
    s_t = df['rot_config'].str[0].map(sgn)
    s_p = df['rot_config'].str[1].map(sgn)
    Jt  = s_t * df['J_rot_targ'] / J_EM
    Jp  = s_p * df['J_rot_proj'] / J_EM
    J0  = df['J_tot_init_sign'] * df['J_tot_init'] / J_EM
    Jorb = J0 - Jt - Jp
    return J_bound(df['gamma'], df['v_inf'], df['b_inf'], Jorb, Jt, Jp,
                   enforce_range=enforce_range)


if __name__ == '__main__':
    import pandas as pd
    df = pd.read_csv('collision_analysis.csv')
    d = df[df.J_bound.notna()].copy()
    meas = (np.sign(d.J_tot_init_sign) * d.J_bound / J_EM).values   # signed measured
    # full-survey fit quality (ignore range, historical number)
    pred_all = np.asarray(J_bound_from_dataframe(d, enforce_range=False))
    r2 = 1 - np.sum((meas-pred_all)**2) / np.sum((meas-meas.mean())**2)
    print(f'All N={len(d)}:           R^2={r2:.4f}  RMSE={np.sqrt(np.mean((meas-pred_all)**2)):.4f} J_EM')
    # in-validity-range fit quality (gamma>=0.1 etc.)
    pred = np.asarray(J_bound_from_dataframe(d, enforce_range=True))
    ok = ~np.isnan(pred)
    r2i = 1 - np.sum((meas[ok]-pred[ok])**2) / np.sum((meas[ok]-meas[ok].mean())**2)
    print(f'In-range N={ok.sum()} (gamma>=0.1): R^2={r2i:.4f}  RMSE={np.sqrt(np.mean((meas[ok]-pred[ok])**2)):.4f} J_EM')
    print(f'Out-of-range rows returned as NaN: {(~ok).sum()}')

    # Non-rotating example (in range): set spins to 0, J_orb = J0
    jb = J_bound(gamma=0.5, v_inf=0.8, b_inf=0.7, J_orb=2.5)
    print(f'Example (gamma=0.5, v_inf=0.8 v_esc, b_inf=0.7, J_orb=2.5 J_EM): '
          f'J_bound = {jb:.3f} J_EM  (retention a={orbital_retention(0.5,0.8,0.7):.3f})')
    # Out-of-range example (planetesimal) -> NaN by design
    print(f'Planetesimal (gamma=0.005): a={orbital_retention(0.005,2.0,0.7)} (NaN: outside validity)')
