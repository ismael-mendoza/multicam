import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

cb_colors_dict = {
    "blue": "#377eb8",
    "orange": "#ff7f00",
    "green": "#4daf4a",
    "purple": "#984ea3",
    "pink": "#f781bf",
    "brown": "#a65628",
    "red": "#e41a1c",
    "yellow": "#dede00",
    "gray": "#999999",
}
CB_COLORS = list(cb_colors_dict.values())
MARKS = ["o", "s", "D", "P", "X"]

LATEX_PARAMS = {
    "cvir": r"$c_{\rm vir}$",
    "t/|u|": r"$T/|U|$",
    "x0": r"$x_{\rm off}$",
    "spin": r"$\lambda_{\rm peebles}$",
    "q": r"$q$",
    "spin_bullock": r"$\lambda_{\rm bullock}$",
    "b_to_a": r"$b/a$",
    "c_to_a": r"$c/a$",
    "r200m/rvir": r"$R_{200m} / R_{\rm vir}$",
    "r500c/rvir": r"$R_{500c} / R_{\rm vir}$",
    "vmax/vvir": r"$V_{\rm max} / V_{\rm vir}$",
    "voff/vvir": r"$V_{\rm off} / V_{\rm vir}$",
    "a2": r"$a_{1/2}$",
    "a4": r"$a_{3/4}$",
    "alpha": r"$\alpha$",
    "mdyn": r"$m(t_{\rm dyn})$",
    "tau_c": r"$\tau_{c}$",
    "alpha_early": r"$\alpha_{\rm early}$",
    "alpha_late": r"$\alpha_{\rm late}$",
}

LATEX_METRICS = {
    "mu": r"$\mu \left( y_{\rm pred} - y_{\rm true} \right)$",
    "med": r"$\mu_{x}'$",
    "sigma_ratio": r"$\sigma_{\rm pred} / \sigma_{\rm true}$",
    "spear": r"$\rho_{\rm spearman} \left(y_{\rm true}, y_{\rm pred} \right)$",
    "rscatter": r"$\frac{\sigma(y_{\rm pred} - y_{\rm true})}{ \sigma(y_{\rm true}) \sqrt{2}}$",
    "mad": r"\rm MAD",
}


def set_rc(
    figsize=(7, 7),
    fontsize=20,
    major_ticksize=8,
    minor_ticksize=5,
    major_tickwidth=1.0,
    minor_tickwidth=0.8,
    ms=6,
    mew=2.0,
    lw=2.0,
    capsize=2.0,
    lgloc="best",
    lgsize="small",
    cmap="Greys",
):
    # relative to fontsize options: 'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large'
    rc_params = {
        # figure
        "figure.figsize": figsize,  # default single axis
        # font.
        "font.family": "sans-serif",
        "font.sans-serif": "Helvetica",
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}",
        "mathtext.fontset": "cm",
        "font.size": fontsize,
        # axes
        "axes.labelsize": "medium",
        "axes.titlesize": "large",
        "axes.axisbelow": True,
        # ticks
        "xtick.labelsize": "small",
        "ytick.labelsize": "small",
        "xtick.major.size": major_ticksize,
        "ytick.major.size": major_ticksize,
        "ytick.minor.size": minor_ticksize,
        "xtick.minor.size": minor_ticksize,
        "xtick.major.width": major_tickwidth,
        "ytick.major.width": major_tickwidth,
        "xtick.minor.width": minor_tickwidth,
        "ytick.minor.width": minor_tickwidth,
        # lines and markers
        "lines.markersize": ms,
        "lines.markeredgewidth": mew,
        "lines.linewidth": lw,
        # errobars
        "errorbar.capsize": capsize,
        # colors
        "axes.prop_cycle": mpl.cycler(color=CB_COLORS),
        # images
        "image.cmap": cmap,
        "image.interpolation": "none",
        # legend
        "legend.loc": lgloc,
        "legend.fontsize": lgsize,
        "legend.frameon": True,
        "legend.framealpha": 0.8,
        "legend.facecolor": "white",
        "legend.edgecolor": "gray",
        # figures
        "figure.autolayout": False,  # same as 'tight_layout'
        # grids
        "axes.grid": True,
        "grid.color": "0.75",  # light gray gridlines
        "grid.linestyle": "-",  # solid gridlines
        "grid.linewidth": 0.5,  # thin gridlines
        "grid.alpha": 1.0,
    }
    mpl.rcParams.update(rc_params)


def draw_histogram(
    ax,
    values,
    n_bins=30,
    bins=None,
    vline="median",
    color="r",
    **hist_kwargs,
):
    ax.hist(
        values,
        bins=bins if bins is not None else n_bins,
        histtype="step",
        color=color,
        **hist_kwargs,
    )

    # add a vertical line.
    if vline == "median":
        ax.axvline(np.median(values), ls="--", color=color)

    elif isinstance(vline, float) or isinstance(vline, int):
        ax.axvline(vline, ls="--", color=color)

    elif vline is None:
        pass

    else:
        raise NotImplementedError(
            f"vline type {type(vline)} is not compatible with current implementation."
        )


def metrics_plot(ax, mval, merr, params, label="", x_bias=0.0, m="o", c="r"):
    """Obtain plots of metrics function given prediction and truth"""
    n_params = len(params)
    assert mval.shape[0] == merr.shape[0] == n_params
    params_latex = [LATEX_PARAMS[par] for par in params]

    # set up axes
    ax.set_xlim(-0.25, n_params)
    ax.set_xticks(np.array(list(range(n_params))))
    ax.set_xticklabels(params_latex)
    for jj in range(n_params):
        label = label if jj == 0 else None
        ax.errorbar(jj + x_bias, mval[jj], yerr=merr[jj], label=label, fmt=m, color=c)
    return ax
