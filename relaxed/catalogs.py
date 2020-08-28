import warnings

import numpy as np
from astropy.table import Table, vstack
from astropy.io import ascii

from pminh import minh

from . import hfilters
from . import parameters

# particle mass (Msun/h), total particles, box size (Mpc/h).
_props = {
    "Bolshoi": (1.35e8, 2048 ** 3, 250),
    "BolshoiP": (1.55e8, 2048 ** 3, 250),
    "MDPL2": (1.51e9, 3840 ** 3, 1000),
}

props = {
    key: {"particle_mass": value[0], "total_particles": value[1], "box_size": value[2]}
    for key, value in _props.items()
}


def intersection(cat, sub_cat):
    """Intersect two catalogs by their id attribute.
    * Returns all rows of cat whose ids are in sub_cat.
    * Full intersection by repeating operation but switching order.
    """
    cat.sort("id")
    sub_cat.sort("id")

    ids = cat["id"]
    sub_ids = sub_cat["id"]

    indx = np.searchsorted(sub_ids, ids)
    indx_ok = indx < len(sub_ids)
    indx_ok[indx_ok] &= sub_ids[indx[indx_ok]] == ids[indx_ok]

    new_cat = cat[indx_ok]

    return new_cat


class HaloCatalog(object):
    def __init__(
        self,
        name,
        cat_path,
        params=None,
        hfilter=None,
        subhalos=False,
        verbose=False,
        label="all halos",
    ):
        """
        * cat_name: Should be one of `Bolshoi / BolshoiP / MDPL2`
        * add_progenitor: filename of summary progenitor table.
        * add_subhalo: add catalog halo properties that depend on their subhalos.
        * labels: useful when plotting (titles, etc.)
        """
        assert name in props, "Catalog name is not recognized."
        assert subhalos is False, "Not implemented subhalo functionality."
        assert cat_path.name.endswith(".minh") or cat_path.name.endswith(".csv")

        self.name = name
        self.cat_path = cat_path
        self.cat_props = props[self.name]
        self.verbose = verbose
        self.subhalos = subhalos
        self.label = label

        self.params = params if params else self.get_default_params()
        self.hfilter = hfilter if hfilter else self.get_default_hfilter()
        assert set(self.hfilter.filters.keys()).issubset(set(self.params))

        self.cat = None  # will be loaded later.

    def __len__(self):
        return len(self.cat)

    @staticmethod
    def get_default_params():
        params1 = ["id", "upid", "mvir", "rvir", "rs", "xoff", "voff"]
        params2 = ["x0", "v0", "cvir", "spin", "q", "vrms"]
        return params1 + params2

    def get_default_hfilter(self):
        default_filters = hfilters.get_default_filters(
            self.cat_props["particle_mass"], self.subhalos
        )
        hfilter = hfilters.HaloFilters(default_filters)
        return hfilter

    def load_cat_csv(self):
        assert self.cat_path.name.endswith(".csv")
        self.cat = ascii.read(self.cat_path, format="csv", fast_reader=True)

    def load_cat_minh(self):
        assert self.cat_path.name.endswith(".minh")
        if self.verbose:
            warnings.warn("Divide by zero errors are ignored, but filtered out.")

        # do filter on the fly, to avoid memory errors.

        with minh.open(self.cat_path) as mcat:

            cats = []

            for b in range(mcat.blocks):
                cat = Table()

                # obtain all params from minh and their values.
                with np.errstate(divide="ignore", invalid="ignore"):
                    for param in self.params:
                        hparam = parameters.get_hparam(param, log=False)
                        values = hparam.get_values_minh_block(mcat, b)
                        cat.add_column(values, name=param)

                # filter to reduce size.
                cat = self.hfilter.filter_cat(cat)
                cats.append(cat)

            self.cat = vstack(cats)

    def save_cat(self, cat_path):
        assert self.cat is not None, "cat must be loaded"
        assert cat_path.suffix == ".csv", "format supported will be csv for now"
        ascii.write(self.cat, cat_path, format="csv")
