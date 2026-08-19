"""
Low-mass A/B/C figure WITH the empirical synestia criterion as star markers.

Companion to gi_cmb_ABC_lowmass.py (target/largest-remnant mass extended down to
MERCURY mass, 0.0553 M_earth) for
  A = Quintana+16 (late-stage)   B = Clement+19 (early instability)
  C = Clement+23 (early instability + inner disk / Mercury component)

Difference from gi_cmb_ABC_lowmass.py: impacts that satisfy the empirical synestia
criterion are drawn as STARS (others as circles).  The original low-mass figure
omitted the Lock+17 hot-CoRoL / HSSL stars because that fit is calibrated only on
Earth-mass bodies.  Here we instead use the joint (non-rotating) synestia criterion
fit to Timpe-Meier + Postema (ANALYSIS_synestia_criterion.md, `_joint_coef.npy`):

    synestia (super-corotation) if   J_bound > A * M_bound^pM * Q_S^pQS   [L_EM]
    (A, pM, pQS) = (1.42, 1.19, -0.40);  M in M_earth, Q_S in MJ/kg.

The AM threshold rises with mass (corotation limit) and falls with heating (Q_S,
which drives the outer structure supercritical -- Caracas & Stewart 2023).  It is
calibrated on M_bound ~ 0.1-2 M_earth, so application between Mercury and Mars mass
(0.055-0.1) is a mild extrapolation (flagged in the caption).

J_bound is the scale-free scaling-law estimate (same eta*L_imp as the CMB pipeline);
Q_S is the modified Leinhardt-Stewart energy (qs_lock.QS_modified).  Colour is still
the estimated L/L_CoRoL^cond.  Selection (all three): embryo-embryo accretionary,
xi>=0.8, gamma>0.1, M_lr>M_Mercury.
"""
import glob, os
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from corotation_limit import L_corot
from Jbound_scaling_law import J_bound_nonrotating_broad
from qs_lock import QS_modified

ME, LEM, JEM = 5.972e24, 3.5e34, 3.61e34
M_MERC = 0.0553                      # Mercury mass [M_earth]
ACC = ['Merge', 'Perfect', 'Graze+Merge']
A_SYN, pQS, pM = np.load('_joint_coef.npy')   # [A, pQS, pM] = 1.42, -0.40, 1.19

# --- Postema+25 Eq 7 / Table 4 : P_CMB(M,L) (robust to low mass) --------------
C_TAB4 = np.array([[153.528, 0.663378, -0.959155], [63.3206, -2.22476, -18.474],
                   [-17.742, -2.63846, -1.60827], [2.61647, -4.23984, -1.74688]])
def _gi(i, M): c = C_TAB4[i]; return c[0]*M**c[1]*np.exp(c[2]*np.exp(-M))
def P_ratio(M, L): return sum(_gi(i, M)*L**i for i in range(4)) / _gi(0, M)

def derive(M, Mt, Mp, Rt, Rp, vimp, b, gamma, vve):
    """(P_CMB/P_norot, L/L_CoRoL, synestia_flag)."""
    Lcor = L_corot(M, type=4)*JEM/LEM                       # condensed Type-4 CoRoL
    mu   = Mt*Mp/(Mt+Mp)
    Limp = mu*vimp*b*(Rt+Rp)
    eta  = J_bound_nonrotating_broad(gamma, vve, b, 1.0)    # scale-free retention law
    Jb   = np.nan_to_num(eta)*Limp/LEM                      # bound AM estimate [L_EM]
    Lret = np.minimum(Jb, Lcor)
    QS   = QS_modified(Mt, Mp, vimp, b, Rt, Rp) / 1e6       # MJ/kg
    syn  = Jb > A_SYN * M**pM * QS**pQS                     # empirical synestia criterion
    return P_ratio(M, Lret), Lret/Lcor, syn

# --- Clement raw loaders (re-extract with the low M_lr floor) -----------------
def load_clement(root, embprefix):
    frames = []
    for f in sorted(glob.glob(os.path.join(root, '*', 'collisions.txt'))):
        frames.append(pd.read_csv(f))
    d = pd.concat(frames, ignore_index=True)
    emb = lambda s: s.str[0].isin(list(embprefix))
    a = d[emb(d.target) & emb(d.projectile) & d.Regime.isin(ACC)].copy()
    a['Mlr'] = a.Mt_Mearth + a.Mp_Mearth - a['m_frag_Mearth']
    a['xi']  = (a.Mlr - a.Mt_Mearth) / a.Mp_Mearth
    a['gam'] = a.Mp_Mearth / a.Mt_Mearth
    return a[(a.Mlr > M_MERC) & (a.xi >= 0.8) & (a.gam > 0.1)].copy()

def derive_clement(c):
    return derive(c.Mlr.values, c.Mt_Mearth.values*ME, c.Mp_Mearth.values*ME,
                  c.Rt_km.values*1e3, c.Rp_km.values*1e3, c['vimp_km/s'].values*1e3,
                  c.b.values, c.gam.values, c['vimp/vesc'].values)

# --- assemble the three datasets ---------------------------------------------
q = pd.read_csv('quintana_collisions.csv')
q = q[(q.Mlr_ME > M_MERC) & (q.xi >= 0.8) & (q.gamma > 0.1)]
qPr, qLo, qS = derive(q.Mlr_ME.values, q.Mt.values, q.Mp.values, q.Rt.values, q.Rp.values,
                      q.Vimp.values, q.b.values, q.gamma.values, q.vve.values)

c1 = load_clement('clement/sims_for_sarah', 'E')
c1Pr, c1Lo, c1S = derive_clement(c1)
c2 = load_clement('clement/sims_for_sarah2_c23', 'EM')
c2Pr, c2Lo, c2S = derive_clement(c2)

panels = [('a', 'Quintana+16', q.Mlr_ME.values, qPr, qLo, qS),
          ('b', 'Clement+19', c1.Mlr.values, c1Pr, c1Lo, c1S),
          ('c', 'Clement+23', c2.Mlr.values, c2Pr, c2Lo, c2S)]

# font sizes
fs1=18 # main labels
fs2=16 # on plot
fs3=12 # small
# ================================ plot ======================================
LBL = dict(fontsize=fs1, color='k')
fig, axs = plt.subplots(3, 1, figsize=(7.6, 15.4), sharex=True)
common = dict(cmap='viridis', vmin=0, vmax=1, edgecolors='k', lw=0.3)
for ax, (lab, name, M, Pr, Lo, S) in zip(axs, panels):
    ax.fill_between([0,3],[.7,.7],[.8,.8],color='grey',alpha=0.25)
    sc = ax.scatter(M[~S], Pr[~S], c=Lo[~S], s=42, marker='o',
                    label=f'planet + disk (N={int((~S).sum())})', **common)
    ax.scatter(M[S], Pr[S], c=Lo[S], s=200, marker='*',
               label=f'synestia (N={int(S.sum())})', **common)
    ax.axhline(np.median(Pr),color='k',ls='--',lw=2)
    ax.axhline(1.0, color='0.5', ls=':', lw=1.2)
    ax.axvline(M_MERC, color='0.6', ls='--', lw=1.0)
    ax.axvline(1.0, color='0.6', ls='--', lw=1.0)
    ax.text(0.058, 0.36, 'Mercury', fontsize=fs3, color='0.45', rotation=90, va='bottom')
    ax.text(1.06, 0.36, 'Earth', fontsize=fs3, color='0.45', rotation=90, va='bottom')
    ax.text(0.06, 0.3, lab, fontsize=fs1, fontweight='bold', va='top') #abc
    ax.text(0.08, 0.3, name, fontsize=fs2, color='0.35', va='top') # study
    ax.set_ylabel(r'$P_{\rm CMB}\,/\,P_{\rm CMB,\,NR}$', **LBL)
    ax.set_ylim(0.25, 1.02); ax.set_xlim(0.045, 2.2); ax.set_xscale('log')
    ax.set_xticks([0.05, 0.1, 0.2, 0.5, 1.0, 2.0])
    ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    ax.tick_params(labelsize=14)
    ax.legend(fontsize=fs3, loc='lower right')
axs[-1].set_xlabel(r'Largest-remnant mass $M_{\rm lr}$  $(M_\oplus)$', **LBL)

fig.subplots_adjust(left=0.135, right=0.97, top=0.99, bottom=0.115, hspace=0.05)
cax = fig.add_axes([0.22, 0.045, 0.56, 0.013])
cb = fig.colorbar(sc, cax=cax, orientation='horizontal')
cb.set_label(r'Estimated $L\,/\,L_{\rm CoRoL}^{\rm cond}$', **LBL)
cb.ax.tick_params(labelsize=14)
fig.savefig('fig_nbody_collisions', dpi=140)
fig.savefig('fig_nbody_collisions.pdf')
for lab, name, M, Pr, Lo, S in panels:
    print(f'{name} ({lab}): N={len(M)}, synestia {int(S.sum())} ({100*S.mean():.0f}%), '
          f'M_lr {M.min():.3f}-{M.max():.3f}, median P/P0={np.median(Pr):.3f}')
print('wrote fig_nbody_collisions.png / .pdf')
