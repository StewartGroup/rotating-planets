#!/usr/bin/env python3
"""Joint synestia criterion in the bound-AM vs mass (CoRoL) plane, pooling
NON-ROTATING Timpe-Meier + Postema (grazing, Mercury+). A synestia forms when
J_bound exceeds a limit that rises with mass and falls with heating (Q_S):
    J_bound > A * M^pM * Q_S^pQS   [L_EM].
The boundary is drawn at low and high Q_S to show the heating shift."""
import os
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from corotation_limit import L_corot, L_canup

J_EM = 3.61e34

HERE = os.path.dirname(os.path.abspath(__file__))   # run from anywhere; data sits beside script
D = pd.read_csv(f"{HERE}/joint_synestia_data.csv")
A, pQS, pM = np.load(f"{HERE}/_joint_coef.npy")

P = D[D.src == 'Postema']; T = D[D.src == 'TM']
fig, ax = plt.subplots(figsize=(4.5, 4))

# TM (non-rotating, grazing) small dots by HSSL label
for lab, sub, c in [('Timpe+23 synestia', T[T.syn==1], '#d1495b'),
                    ('Timpe+23 planet+disk',                 T[T.syn==0], '#4a90d9')]:
    ax.scatter(sub.M, sub.J, s=10, marker='s', facecolor=c, alpha=0.45, edgecolor='k', linewidth=0.1, label=f'{lab} (n={len(sub)})')

# Postema markers by label
for lab, sub, mk, c in [('Postema+26 synestia', P[P.syn==1], '^', '#d1495b'),
                        ('Postema+26 planet+disk',       P[P.syn==0], 'o', '#4a90d9')]:
    ax.scatter(sub.M, sub.J, s=20, marker=mk, facecolor=c, edgecolor='k',
               linewidth=0.5, alpha=0.95, label=f'{lab} (n={len(sub)})')

MM = np.logspace(np.log10(0.09), np.log10(2.1), 100)
LL = L_corot(MM, type=4)
ttt = r'$L_{\rm CoRoL}^{\rm hot}$'
ax.plot(MM,LL,'k-',label=r'$L_{\rm CoRoL}^{\rm cond}$')
for QS, ls, lab in [(3.0, '--', r'$Q_S=3$ MJ/kg (less heating)'),
                    (15.0, ':',  r'$Q_S=15$ MJ/kg (more heating)')]:
    ax.plot(MM, A * MM**pM * QS**pQS, color='k', ls=ls, lw=2,
            label=f'{ttt}, {lab}')

ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel(r'$M_{\rm bnd}$  [$M_\oplus$]', fontsize=12)
ax.set_ylabel(r'$L_{\rm bnd}$  [$L_{\rm EM}$]', fontsize=12)
#ax.set_title('Joint synestia criterion (non-rotating TM + Postema)\n'
#             r'$J_{\rm bound} > %.2f\,M^{%.2f}\,Q_S^{%.2f}$   (AM $\uparrow$ mass, $\downarrow$ heating)'
#             % (A, pM, pQS), fontsize=11)
ax.legend(fontsize=7, loc='upper left', framealpha=0.95)
ax.grid(True, which='both', ls=':', alpha=0.3)
fig.tight_layout()
fig.savefig(f"{HERE}/fig_synestia_criteria.pdf", bbox_inches='tight')
fig.savefig(f"{HERE}/fig_synestia_criteria.png", dpi=150, bbox_inches='tight')
print(f"wrote fig_synestia_criteria.pdf / .png   (J_crit = {A:.2f} M^{pM:.2f} QS^{pQS:.2f})")
