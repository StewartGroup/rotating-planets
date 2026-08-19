"""
Exploratory assembly for a graze-and-merge / ejecta-based bound-AM scaling.

Idea under test (S. Stewart): the retained (bound) AM is the total merged AM J_0
minus the AM carried off by UNBOUND EJECTA (not disk).  So instead of fitting the
retention eta = J_b/J_0 directly against geometry, ask whether the *AM lost fraction*

        1 - eta = J_ej / J_0

tracks the *mass lost fraction*

        f_ej = M_ej / M_tot

with a geometric "specific-AM enhancement"

        Lambda = (1 - eta) / f_ej          (>1 => ejecta carries more than its share of AM)

Datasets (both non-rotating):
  - Timpe-Meier NN: M_ej, J_ej given directly (merged events = J_bound defined).
  - Postema: ejecta = total - bound;  M_ej = (M_t+M_i) - M_bnd,  J_ej = Lztot - Lzbnd.
    outcomemarker: o=accretion, P=graze-and-merge/partial, x=hit-and-run(excluded).
"""
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------------ load TM
def load_tm():
    t = pd.read_csv('collision_analysis.csv')
    nn = t[(t.rot_config == 'NN') & (t.J_bound.notna())].copy()
    d = pd.DataFrame({
        'eta'   : nn.J_bound / nn.J_tot_init,
        'f_ej'  : nn.M_ej / nn.M_tot,
        'am_ej' : nn.J_ej / nn.J_tot_init,          # measured AM-lost fraction
        'gamma' : nn.gamma,
        'vimp'  : np.sqrt(nn.v_inf.values**2 + 1.0),
        'b'     : nn.b_imp,
        'sinth' : np.sin(np.radians(nn.theta_imp.values)),
        'Mt'    : nn.M_targ,
        'xi'    : nn.accretion_efficiency,
        'ctype' : nn.collision_type,
    })
    d['src'] = 'TimpeMeier'
    return d

# ------------------------------------------------------------------ load Postema
def load_postema():
    p = pd.read_csv('Postema-giant-impacts/Manuscript_data_table_clean.csv')
    p = p[p.outcomemarker.isin(['o', 'P'])].copy()          # merged only, drop hit-and-run
    Mtot = p['M_t (Earth)'] + p['M_i (Earth)']
    Mej  = Mtot - p['Mbound (Earth)']
    d = pd.DataFrame({
        'eta'   : p['Lzbnd (LEM)'] / p['Lztot (LEM)'],
        'f_ej'  : Mej / Mtot,
        'am_ej' : (p['Lztot (LEM)'] - p['Lzbnd (LEM)']) / p['Lztot (LEM)'],
        'gamma' : p['M_i (Earth)'] / p['M_t (Earth)'],
        'vimp'  : p['vi/v_esc'],
        'b'     : p['b'],
        'sinth' : p['b'],
        'Mt'    : p['M_t (Earth)'],
        'xi'    : (p['Mbound (Earth)'] - p['M_t (Earth)']) / p['M_i (Earth)'],
        'theta' : p['theta (deg)'],
        'theta_crit': p['theta_crit'],
        'marker': p['outcomemarker'],
    })
    d['src'] = 'Postema'
    return d

def assemble():
    tm, po = load_tm(), load_postema()
    d = pd.concat([tm, po], ignore_index=True)
    # keep physical, non-degenerate mergers with real ejecta and defined eta
    d = d[(d.eta > 0) & (d.eta <= 1.05) & (d.f_ej > 0)].copy()
    d['Lambda'] = (1 - d.eta) / d.f_ej
    # geometric graze-and-merge flag: grazing (theta>theta_crit for Postema),
    #  or high impact parameter b for TM
    grazing = np.zeros(len(d), bool)
    pmask = d.src == 'Postema'
    grazing[pmask.values] = (d.loc[pmask, 'theta'] > d.loc[pmask, 'theta_crit']).values
    tmask = (d.src == 'TimpeMeier').values
    grazing[tmask] = (d.loc[d.src == 'TimpeMeier', 'b'] > 0.7).values
    d['grazing'] = grazing
    return d

if __name__ == '__main__':
    d = assemble()
    print(f"total merged events: {len(d)}  (TM {sum(d.src=='TimpeMeier')}, Postema {sum(d.src=='Postema')})")
    print("\n-- ejecta mass/AM sanity --")
    print(f"  f_ej   median {d.f_ej.median():.4f}  range {d.f_ej.min():.4f}-{d.f_ej.max():.3f}")
    print(f"  1-eta  median {(1-d.eta).median():.4f}  range {(1-d.eta).min():.4f}-{(1-d.eta).max():.3f}")
    print(f"  Lambda=(1-eta)/f_ej  median {d.Lambda.median():.2f}  IQR {d.Lambda.quantile(.25):.2f}-{d.Lambda.quantile(.75):.2f}")

    # correlation of AM-lost with mass-lost (log space)
    m = d[(d.f_ej > 1e-4) & ((1-d.eta) > 1e-4)]
    r = np.corrcoef(np.log10(m.f_ej), np.log10(1-m.eta))[0,1]
    print(f"\n  corr[ log f_ej , log(1-eta) ] = {r:+.3f}   (N={len(m)})")
    # slope of log(1-eta) ~ a*log(f_ej)+b
    A = np.polyfit(np.log10(m.f_ej), np.log10(1-m.eta), 1)
    print(f"  log(1-eta) = {A[0]:.3f} log(f_ej) + {A[1]:.3f}   => (1-eta) ~ f_ej^{A[0]:.2f}")

    # Lambda vs geometry
    print("\n-- Lambda vs impact parameter b (grazing enhancement) --")
    for lo, hi in [(0,0.3),(0.3,0.5),(0.5,0.7),(0.7,0.85),(0.85,1.01)]:
        s = d[(d.b>=lo)&(d.b<hi)]
        if len(s): print(f"  b in [{lo:.2f},{hi:.2f}): N={len(s):3d}  Lambda_med={s.Lambda.median():5.2f}  f_ej_med={s.f_ej.median():.4f}")

    d.to_csv('ejecta_AM_explore.csv', index=False)
    print("\nwrote ejecta_AM_explore.csv")

    # ============================================================ PLOTS
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    cmap = plt.cm.viridis

    # -- Panel 1: AM-lost fraction vs mass-lost fraction (the core hypothesis)
    for src, mk in [('TimpeMeier','o'), ('Postema','s')]:
        s = d[d.src==src]
        sc = ax[0].scatter(s.f_ej, 1-s.eta, c=s.b, cmap=cmap, vmin=0, vmax=1,
                           marker=mk, s=38, edgecolor='k', linewidth=0.3, label=src)
    xx = np.logspace(-4, 0, 50)
    for L, ls in [(1,':'),(2,'--'),(5,'-.')]:
        ax[0].plot(xx, np.clip(L*xx,0,1), 'k', ls=ls, lw=1, alpha=.6, label=f'$\\Lambda$={L}')
    ax[0].set(xscale='log', yscale='log', xlim=(3e-4,1), ylim=(3e-4,1),
              xlabel='mass-lost fraction  $f_{ej}=M_{ej}/M_{tot}$',
              ylabel='AM-lost fraction  $1-\\eta = J_{ej}/J_0$')
    ax[0].set_title('AM lost tracks mass lost?')
    ax[0].legend(fontsize=7, loc='lower right')
    cb = fig.colorbar(sc, ax=ax[0]); cb.set_label('impact parameter $b=\\sin\\theta$')

    # -- Panel 2: specific-AM enhancement Lambda vs impact parameter (accretionary set)
    #    per-point Lambda is very scattered; the trend is in the BINNED MEDIANS.
    ac = d[(d.xi > 0.2) & (d.f_ej > 1e-4) & ((1-d.eta) > 1e-4)].copy()
    for src, mk in [('TimpeMeier','o'), ('Postema','s')]:
        s = ac[ac.src==src]
        ax[1].scatter(s.b, s.Lambda, c=s.vimp, cmap='plasma', vmin=1, vmax=3,
                      marker=mk, s=26, edgecolor='k', linewidth=0.25, alpha=.55, label=src)
    ax[1].axhline(1, color='k', ls=':', lw=1)
    # binned medians (the trend)
    edges = np.array([0,0.3,0.5,0.7,0.85,1.01]); cen=[]; med=[]
    for lo,hi in zip(edges[:-1],edges[1:]):
        s=ac[(ac.b>=lo)&(ac.b<hi)]
        if len(s): cen.append(s.b.median()); med.append(s.Lambda.median())
    ax[1].plot(cen, med, 'D-', color='k', ms=9, lw=1.8, zorder=6,
               mfc='white', mec='k', label='binned median $\\Lambda$')
    # fitted Lambda(b)=A(1+b)^m
    Xl = np.column_stack([np.log(1+ac.b.values), np.ones(len(ac))])
    cl,*_ = np.linalg.lstsq(Xl, np.log(ac.Lambda.values), rcond=None)
    mL, Al = cl[0], np.exp(cl[1])
    R2L = 1 - np.sum((np.log(ac.Lambda)-Xl@cl)**2)/np.sum((np.log(ac.Lambda)-np.log(ac.Lambda).mean())**2)
    bb = np.linspace(0,1,50)
    ax[1].plot(bb, Al*(1+bb)**mL, 'r-', lw=2.2, zorder=7,
               label=f'$\\Lambda(b)={Al:.1f}(1+b)^{{{mL:.1f}}}$')
    ax[1].set(yscale='log', ylim=(0.1, 200),
              xlabel='impact parameter $b=\\sin\\theta$',
              ylabel='$\\Lambda=(1-\\eta)/f_{ej}$  (specific-AM enhancement)')
    ax[1].set_title(f'$\\Lambda$ rises with grazing (per-point $R^2_{{\\log}}$={R2L:.2f}: mostly scatter)')
    sc2 = ax[1].collections[0]
    cb2 = fig.colorbar(sc2, ax=ax[1]); cb2.set_label('$v_{imp}/v_{esc}$')
    ax[1].legend(fontsize=6.5, loc='upper left')

    # -- Panel 3: candidate MULTIPLICATIVE model for the AM lost to ejecta:
    #      J_ej/J_0 = 1-eta = A * f_ej^s * (1+b)^m
    #    fit in log space (robust; no linear-model failure tail), then eta = 1 - that.
    #    RESTRICT to accretionary/graze-merge regime (xi>0.2): drops the erosive
    #    disruption family (negative xi, tiny fast grazing impactor) that is a
    #    separate regime.  Parsimonious mass-only power law (adding v_imp/gamma only
    #    marginally raises R2 and is collinear with f_ej -> not defensible).
    mm = d[(d.f_ej > 1e-4) & ((1-d.eta) > 1e-4) & (d.xi > 0.2)].copy()
    y = np.log((1 - mm.eta).values)
    X = np.column_stack([np.log(mm.f_ej.values), np.ones(len(mm))])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    s_exp, lnA = coef
    A = np.exp(lnA)
    d['amloss_pred'] = A * d.f_ej**s_exp
    d['eta_pred']    = np.clip(1 - d.amloss_pred, 0, 1)
    # report power-law quality on the loss itself
    yhat = X @ coef
    R2log = 1 - np.sum((y-yhat)**2)/np.sum((y-y.mean())**2)
    # plot only the fitted (accretionary) regime; disruptions shown as grey x
    fit_mask = d.index.isin(mm.index)
    dis = d[(~fit_mask) & (d.xi <= 0.2)]
    ax[2].scatter(np.clip(1-dis.amloss_pred,0,1), dis.eta, marker='x', c='0.6',
                  s=30, label='erosive/disrupt ($\\xi\\!<\\!0.2$)')
    for src, mk in [('TimpeMeier','o'), ('Postema','s')]:
        s = d[fit_mask & (d.src==src)]
        ax[2].scatter(s.eta_pred, s.eta, c=s.b, cmap=cmap, vmin=0, vmax=1,
                      marker=mk, s=38, edgecolor='k', linewidth=0.3, label=src)
    ax[2].plot([0,1],[0,1],'k-',lw=1)
    ax[2].set(xlim=(0,1.02), ylim=(0,1.02),
              xlabel='predicted $\\eta = 1-A\\,f_{ej}^{s}$',
              ylabel='measured $\\eta = J_b/J_0$')
    resid = (d.eta - d.eta_pred)[fit_mask]
    ax[2].set_title(f'mass-only: $1-\\eta={A:.2f}\\,f_{{ej}}^{{{s_exp:.2f}}}$   '
                    f'$R^2_{{log}}$={R2log:.2f}  RMS$_\\eta$={np.sqrt((resid**2).mean()):.3f}')
    ax[2].legend(fontsize=6.5, loc='upper left')
    print(f"\n-- parsimonious loss model (xi>0.2)  1-eta = A f_ej^s --")
    print(f"   A={A:.3f}  s={s_exp:.3f}   R2(log-loss)={R2log:.3f}  "
          f"RMS(eta,fit)={np.sqrt((resid**2).mean()):.3f}  N={len(mm)}")

    fig.tight_layout()
    fig.savefig('ejecta_AM_explore.png', dpi=140)
    print("wrote ejecta_AM_explore.png")
