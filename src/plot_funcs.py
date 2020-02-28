"""
Plotting functions that are useful for visualizing things like correlations.
"""

import numpy as np
import matplotlib.pyplot as plt
from src.frames import params


def general_settings(ax, title='', xlabel=None, ylabel=None, xlabel_size=18, ylabel_size=18, legend_label=None,
                     legend_size=18, title_size=18):

    ax.set_title(title, fontsize=title_size)

    if xlabel is not None and ylabel is not None:
        ax.set_xlabel(xlabel, size=xlabel_size)
        ax.set_ylabel(ylabel, size=ylabel_size)

    if legend_label:
        ax.legend(loc='best', prop={'size': legend_size})


def histogram(cat, param, ax, bins=30, histtype='step', color='r', legend_label=None, hist_kwargs=None,
              **general_kwargs):
    if hist_kwargs is None:
        hist_kwargs = {}

    values = param.get_values(cat)
    ax.hist(values, bins=bins, histtype=histtype, color=color, label=legend_label, **hist_kwargs)

    general_settings(ax)


def binning3d_mass(cat, param1, param2, ax, mass_decades=range(11, 15), legend_size=18, **plot_kwargs):
    """
    * plot_kwargs are additional keyword arguments to pass into the plotting_func
    * mods: lambda functions that modify plotting arrays, e.g. lambda x: np.log10(x)
    """
    mass_bins = [(x, y) for x, y in zip(mass_decades, mass_decades[1:])]
    colors = ['b', 'r', 'g'] 
    for mass_bin, color in zip(mass_bins, colors): 
        log_mvir = params.Param('mvir', log=True).get_values(cat)
        mmask = (log_mvir > mass_bin[0]) & (log_mvir < mass_bin[1])
        mcat = cat[mmask]
        label = "$" + str(mass_bin[0]) + "< M_{\\rm vir} <" + str(mass_bin[1]) + "$"
        scatter_binning(mcat,
                        param1, param2,
                        color=color, legend_label=label, ax=ax, **plot_kwargs)
    
    ax.legend(prop={"size": legend_size}, loc='best')


def scatter_binning(cat, param1, param2, ax, xlabel=None, ylabel=None, nxbins=10, color='r', no_bars=False,
                    show_lines=False, show_bands=False, legend_label=None, **general_kwargs):

    x = param1.get_values(cat)
    y = param2.get_values(cat)

    xs = np.linspace(np.min(x), np.max(x), nxbins)
    xbbins = [(xs[i], xs[i+1]) for i in range(len(xs)-1)]

    masks = [((xbbin[0] < x) & ( x < xbbin[1])) for xbbin in xbbins]
    binned_x = [x[mask] for mask in masks]
    binned_y = [y[mask] for mask in masks]

    xmeds = [np.median(xbin) for xbin in binned_x]
    ymeds = [np.median(ybin) for ybin in binned_y]

    xqs = np.array([[xmed - np.quantile(xbin, 0.25), np.quantile(xbin, 0.75) - xmed] for (xmed, xbin)
                    in zip(xmeds, binned_x)]).T
    yqs = np.array([[ymed - np.quantile(ybin, 0.25), np.quantile(ybin, 0.75) - ymed] for (ymed, ybin)
                    in zip(ymeds, binned_y)]).T

    if not no_bars:
        ax.errorbar(xmeds, ymeds, xerr=xqs, yerr=yqs, fmt='o--', capsize=10, color=color, label=legend_label)
    else:
        ax.errorbar(xmeds, ymeds, xerr=xqs, fmt='o-', color=color, label=legend_label, capsize=10)

    y1 = np.array([np.quantile(ybin, 0.25) for ybin in binned_y])
    y2 = np.array([np.quantile(ybin, 0.75) for ybin in binned_y])

    if show_lines:
        ax.plot(xmeds, y1, '--', color=color)
        ax.plot(xmeds, y2, '--', color=color)

    if show_bands:
        ax.fill_between(xmeds, y1, y2, alpha=0.2, linewidth=0.001, color=color)

    general_settings(ax, xlabel=xlabel, ylabel=ylabel, legend_label=legend_label, **general_kwargs)


