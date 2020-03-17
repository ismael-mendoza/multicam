import astropy
from astropy.table import Table
from astropy.io import ascii
import numpy as np
from typing import List

from src.frames import params
from src.frames import filters

# particle mass (Msun/h), total particles, box size (Mpc/h).
catalog_properties = {
    'Bolshoi': (1.35e8, 2048**3, 250),
    'BolshoiP': (1.55e8, 2048**3, 250),
    'MDPL2': (1.51e9, 3840**3, 1000)
}


class HaloCatalog(object):

    def __init__(self, filepath, catalog_name, subhalos=False, base_filters=None,
                 params_to_include: List[str] = None, verbose=False, catalog_label='all halos'):
        """

        :param filepath:
        :param catalog_name: Should be one of `Bolshoi / BolshoiP / MDPL2`
        :param base_filters:
        """
        assert catalog_name in catalog_properties, "Catalog name is not recognized."
        assert subhalos is False, "Not implemented subhalo functionality."

        self.filepath = filepath
        self.catalog_name = catalog_name
        self.verbose = verbose
        self.subhalos = subhalos
        self.particle_mass, self.total_particles, self.box_size = catalog_properties[self.catalog_name]
        self.catalog_label = catalog_label  # for use in things like legends.

        # name of all params that will be needed for filtering and
        # params we actually want to have at the end in the output catalog.
        self.param_names = params.param_names
        self.params_to_include = params_to_include if params_to_include else params.default_params_to_include

        self._cfilters = base_filters if base_filters else filters.get_default_base_filters(self.particle_mass,
                                                                                           self.subhalos)
        if not set(self._cfilters.keys()).issubset(set(self.param_names)):
            raise ValueError("filtering will fail since not all params are in self.param_names,"
                             "need to update params.py")

        # will be potentially defined later.
        self.use_generator = None
        self._bcat = None  # base cat.
        self._cat = None  # cat to actually return.

        # ToDo: For the generator case we want to returned a modified generator with the filtered values.
        # ToDo: Make everything work together property if not default filters or params to include.
        # ToDo: Create a filtered_from_cat method. (general filtering not only relaxed)
        # ToDo: Delay obtaining filters so that we can use parameters of catalog in the user-defined filters.
        #  (Necessary?)
        # ToDo: Change everything in place or create copy for every catalog (relaxed, mass bins, etc.)?
        #       for now everything is copy.

    def get_cat(self):
        return self._cat

    def get_cfilters(self):
        return self._cfilters

    def with_filters(self, myfilters, catalog_label='filtered catalog'):
        self._cat = self._filter_cat(self._cat, myfilters)
        self.catalog_label = catalog_label

    def with_relaxed_filters(self, relaxed_name=None):
        self.with_filters(filters.get_relaxed_filters(relaxed_name), catalog_label=f"{relaxed_name} relaxed")

    def load_base_cat(self, use_generator=False, bcat=None):
        """
        This function is used to set the cat attribute in the hcat to the catalog so it can be used in the future.
        :param use_generator:
        :param bcat:
        :return:
        """
        assert use_generator is False, "Not implemented this functionality yet, for now just return full catalog."
        assert self._bcat is None, "Overriding catalog that is already created. (probably wasteful)"

        self.use_generator = use_generator
        if not bcat:
            self._bcat = self._load_cat()  # could be a generator or an astropy.Table object.

        else:
            self._bcat = bcat

        self._cat = self._bcat  # filtering will create a copy later if necessary.

    def _get_cat_generator(self):
        """
        This will eventually contain all the complexities of Phil's code and return a generator to access
        the catalog in chunks. For now it offers the chunkified version of my catalog how I've been doing it.
        :return:
        """

        # 100 Mb chunks of maximum memory in each iteration.
        # this returns a generator.
        return ascii.read(self.filepath, format='csv', guess=False,
                          fast_reader={'chunk_size': 100 * 1000000, 'chunk_generator': True})

    def _load_cat(self):
        """
        Return either the catalog as a table or as a generator. Should only be called by set_cat.

        NOTE: We filter using the cfilters (cat filters).
        :return:
        """
        gcats = self._get_cat_generator()

        if self.use_generator:
            return gcats

        else:
            if self.verbose:
                print("Ignoring dividing by zero and invalid errors that should be filtered out anyways.")

            # actually extract the data from gcats and read it into memory.
            # do filtering on the fly so don't actually ever read unfiltered catalog.
            cats = []

            for i, cat in enumerate(gcats):
                new_cat = Table()

                # ignore warning of possible parameters that are divided by zero, this will be filtered out later.
                with np.errstate(divide='ignore', invalid='ignore'):
                    for param in self.param_names:
                        new_cat.add_column(self._get_not_log_value(cat, param), name=param)

                new_cat = self._filter_cat(new_cat, self._cfilters, use_include_params=True)

                cats.append(new_cat)

                if self.verbose:
                    if i % 10 == 0:
                        print(i)

            return astropy.table.vstack(cats)

    def _filter_cat(self, cat, myfilters, use_include_params=False):
        """
        * Do all the appropriate filtering required when not reading the generator expression, in particular cat is
        assumed to contain all the parameter in param_names before value filters are applied.

        * Not all parameters are actually required once filtering is complete so they can be (optionally)
        removed according to self.params_to_include.

        NOTE: All filters assumed no logging has been done on the raw catalog columns.

        :param cat:
        :param myfilters:
        :param use_include_params: whether to remove all parameters that are not in `self.params_to_include`
        :return:
        """
        for param_name, myfilter in myfilters.items():
            cat = cat[myfilter(self._get_not_log_value(cat, param_name))]

        if use_include_params:
            cat = cat[self.params_to_include]

        return cat

    @staticmethod
    def _get_not_log_value(cat, key):
        """
        Only purpose is for the filters.
        :param cat:
        :param key:
        :return:
        """
        return params.Param(key, log=False).get_values(cat)

    @classmethod
    def create_filtered_from_base(cls, old_hcat, myfilters, catalog_label='filtered cat'):
        assert old_hcat.get_cat() is not None, "Catalog of old_hcat should already be set."
        assert set(myfilters.keys()).issubset(set(old_hcat.cat.colnames)), "This will fail because the " \
                                                                           "cat of old_hcat does " \
                                                                           "not contain filtered parameters."

        new_hcat = cls(old_hcat.filename, old_hcat.catalog_name, subhalos=old_hcat.subhalos,
                       base_filters=old_hcat.get_filters(), catalog_label=catalog_label,
                       params_to_include=old_hcat.params_to_include)

        assert set(myfilters.keys()).issubset(set(old_hcat.cat.colnames)), "This will fail because the " \
                                                                           "final catalog of old_hcat does " \
                                                                           "not contain filtered parameters."
        new_hcat.load_base_cat(use_generator=False, bcat=old_hcat.get_cat())
        new_hcat.with_filters(myfilters, catalog_label=catalog_label)
        return new_hcat

    @classmethod
    def create_relaxed_from_base(cls, old_hcat, relaxed_name):
        return cls.create_filtered_from_base(old_hcat, filters.get_relaxed_filters(relaxed_name),
                                             catalog_label=f"{relaxed_name} relaxed")

    @classmethod
    def create_from_saved_cat(cls, cat_file, *args, **kwargs):
        """
        Create catalog from saved (smaller/filtered) cat to reduce waiting time of having to filter everytime, etc.
        :param cat_file: The file location specified as Path object and save using
        :return:
        """
        hcat = cls(*args, **kwargs)
        cat = ascii.read(cat_file, format='csv')
        hcat.cat = cat
        return hcat
