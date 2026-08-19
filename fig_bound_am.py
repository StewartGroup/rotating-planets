"""
Clean two-panel slide figure for the ejecta-based J_bound retention law.
  (A) AM lost tracks mass lost:  1-eta vs f_ej.
  (B) the law works:  predicted vs measured eta.
"""
import numpy as np, matplotlib.pyplot as plt
from explore_ejecta_AM import assemble

A, m, s = 4.362941299820175, 1.274173427265838, 1.101167386820984   # eta = 1 - A(1+b)^m f_ej^s

d   = assemble()
acc = d[(d.xi > 0.2) & (d.f_ej > 1e-4) & ((1 - d.eta) > 1e-4)].copy()   # accretionary incl. graze-merge
acc['eta_pred'] = np.clip(1 - A*(1+acc.b)**m * acc.f_ej**s, 0, 1)
rms = np.sqrt(((acc.eta - acc.eta_pred)**2).mean())

plt.rcParams.update({'font.size': 15, 'axes.labelsize': 16, 'axes.titlesize': 17})
fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
cmap = plt.cm.viridis

# ---- (A) AM lost vs mass lost
for src, mk, lab in [('TimpeMeier', 'o', 'Timpe+23'), ('Postema', 's', 'Postema+26')]:
    g = acc[acc.src == src]
    sc = ax[0].scatter(g.f_ej, 1 - g.eta, c=g.b, cmap=cmap, vmin=0, vmax=1,
                       marker=mk, s=45, edgecolor='k', linewidth=0.3, label=lab)
xx = np.logspace(-4, 0, 50)
ax[0].plot(xx, np.clip(8.77*xx**1.16, 0, 1), 'k--', lw=2, label=r'$1-\eta\propto f_{\rm ej}^{1.2}$')
ax[0].set(xscale='log', yscale='log', xlim=(3e-4, 1), ylim=(3e-4, 1.2),
          xlabel=r'mass lost   $f_{\rm ej}=1-M_{\rm lr}/M_{\rm tot}$',
          ylabel=r'AM lost   $1-\eta=L_{\rm ej}/L_0$')
#ax[0].set_title('AM lost tracks mass lost', fontweight='bold')
ax[0].legend(fontsize=12, loc='upper left', framealpha=0.9)
ax[0].text(.6,6.e-4,'a',fontsize=16,fontweight='bold',va='top') #ab
cb = fig.colorbar(sc, ax=ax[0]); cb.set_label(r'impact param. $b=\sin\theta$')

# ---- (B) predicted vs measured eta
for src, mk in [('TimpeMeier', 'o'), ('Postema', 's')]:
    g = acc[acc.src == src]
    ax[1].scatter(g.eta_pred, g.eta, c=g.b, cmap=cmap, vmin=0, vmax=1,
                  marker=mk, s=45, edgecolor='k', linewidth=0.3)
ax[1].plot([0, 1], [0, 1], 'k-', lw=1.5)
ax[1].set(xlim=(0, 1.03), ylim=(0, 1.03),
          xlabel=r'predicted $\eta = 1-A(1+b)^m f_{\rm ej}^{s}$',
          ylabel=r'measured $\eta = L_{\rm bnd}/L_0$')
#ax[1].set_title(f'retention law   (RMS $={rms:.2f}$)', fontweight='bold')
ax[1].text(0.05, 0.93, f'$N={len(acc)}$ accretionary\nimpacts',
           transform=ax[1].transAxes, fontsize=12, va='top')
ax[1].text(.98,.08,'b',fontsize=16,fontweight='bold',va='top') #ab

fig.tight_layout()
fig.savefig('fig_bound_am.png', dpi=150, bbox_inches='tight')
fig.savefig('fig_bound_am.pdf')
print(f'wrote fig_bound_am.png   (N={len(acc)}, RMS(eta)={rms:.3f})')
