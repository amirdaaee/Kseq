import numpy as np
import pandas as pd
from keras import utils
import copy
import abc
import inspect
import traceback


class SeqGen(utils.Sequence):
    dataset: pd.DataFrame

    def __init__(self,
                 dataset,
                 input_length,
                 output_length,
                 feature_columns,
                 target_columns,
                 batch_size=64,
                 sort=1,
                 inbatch_suffle=False,
                 verbosity=False):
        """
        Parameters
        ----------
        dataset : list[pd.DataFrame] or pd.DataFrame
        input_length , output_length: int
            must be >=0
        feature_columns , target_columns: list
            values must be in the `dataset.columns`
        batch_size: int
            -1: length equal to whole data
        sort: {1, 0, -1}
            sort `dataset` based on the index
            1: ascending sort (default)
            0: Do not perform sorting
            -1: descending sort
        inbatch_suffle: bool
            whether shuffle data `on_epoch_end`
        verbosity: bool
        """

        if verbosity:
            print('TS generator initializing...')
        # ...................... preparing args
        if type(dataset) == pd.DataFrame:
            dataset = [dataset]
        self.dataset = copy.deepcopy(dataset)
        self.input_length, self.output_length = input_length, output_length
        self.feature_col, self.target_col = feature_columns, target_columns
        self.shuffle = inbatch_suffle
        # ......................
        if sort != 0:
            sort = {1: True, -1: False}[sort]
            for c, ds in enumerate(self.dataset):
                self.dataset[c] = ds.sort_index(ascending=sort)
            if verbosity:
                print('sorting : ascending', sort)
        else:
            if verbosity:
                print("sorting : no")
        # ..................................................
        ts_len = self.input_length + self.output_length
        tmp_array = []
        for c, ds in enumerate(self.dataset):
            m = len(ds) - ts_len + 1
            tmp_array.append(np.array([np.arange(i, ts_len + i) for i in range(m)]))
            if c == 0:
                continue
            last_index = tmp_array[-2][-1, -1]
            tmp_array[-1] += (last_index + 1)

        all_ts = np.vstack(tmp_array).astype(np.uint32)
        self.x_ts, self.y_ts = np.hsplit(all_ts, (self.input_length,))
        self.sample_n = len(all_ts)
        self.batch_size = batch_size
        # ....................................................
        conc_dataset = pd.concat(self.dataset, axis=0)
        self.X = conc_dataset[self.feature_col].values
        self.Y = conc_dataset[self.target_col].values
        # ....................................................
        self.__reset_batch_index__()
        self.mode = 'batch'
        if verbosity:
            print(self)

    # ====================================== Modes
    def __len__(self):
        if self.mode == 'batch':
            return self.batch_n
        else:
            return self.sample_n

    def __getitem__(self, idx):
        """
        Parameters
        ----------
        idx : int or list[int] or slice
        """
        try:
            return self.__getitem__mode__(idx)
        except Exception as e:
            traceback.print_tb(e.__traceback__)
            print('index', idx)
            raise Exception(e)

    # -------------- abs mode
    def __getitem__abs__(self, idx):
        ds = [self.X, self.Y]
        ts = [self.x_ts, self.y_ts]
        fs = [self.xfn, self.yfn]
        r = [None, None]
        for c, d, t, f in zip([0, 1], ds, ts, fs):
            i = t[idx, :]
            r[c] = d[i, :].copy()
            if r[c].size == 0:
                r[c] = None
                continue
            if r[c].ndim == 2:
                r[c] = r[c][np.newaxis, :, :]
            if f is not None:
                args_n = len(inspect.getfullargspec(f).args)
                if args_n == 1:
                    r[c] = f(r[c])
                elif args_n == 2:
                    r[c] = f(self, r[c])
                else:
                    raise Exception('xfn or yfn arguments not understood')
        return r

    # -------------- batch mode
    def __getitem__batch__(self, idx):  # needs more works!
        assert idx.__class__ == int, 'in batch mode, index has to be a single int'
        assert 0 <= idx < self.batch_n, "batch {} doesn't exist".format(idx)
        idx = self.meta_index[idx * self.batch_size:(idx + 1) * self.batch_size]
        return self.__getitem__abs__(idx)

    # ====================================== other functions
    def __str__(self):
        s = ''
        s += 'input_length :\t{}\n'.format(self.input_length)
        s += 'output_length :\t{}\n'.format(self.output_length)
        s += 'feature_col :\t{}\n'.format(self.feature_col)
        s += 'target_col :\t{}\n'.format(self.target_col)
        s += '.' * 20
        s += '\n'
        s += 'sample_n :\t{}\n'.format(self.sample_n)
        s += 'batch_size :\t{}\n'.format(self.batch_size)
        s += 'batch_n :\t{}\n'.format(self.batch_n)
        s += '.' * 20
        s += '\n'
        s += 'mode :\t{}\n'.format(self.mode)
        s += '-' * 50
        return s

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.meta_index)

    def __reset_batch_index__(self):
        self.meta_index = np.arange(self.sample_n, dtype=np.uint32)

    # ====================================== properties
    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, batch_size):
        if batch_size == -1:
            self._batch_size = self.sample_n
        elif batch_size >= 1:
            self._batch_size = int(batch_size)
        else:
            raise (Exception("batch_size should be -1 or >=1. {} is passed".format(batch_size)))
        # ......................
        self.batch_n = int(np.ceil(self.sample_n / self._batch_size))

    # ====================================== mode
    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, mode):
        """
        Parameters
        ----------
        mode : {'batch', 'abs'}
        """
        if mode == 'batch':
            self.__getitem__mode__ = self.__getitem__batch__
        elif mode == 'abs':
            self.__getitem__mode__ = self.__getitem__abs__
        else:
            raise Exception("mode didn't understood : {}".format(mode))
        self._mode = mode

    # ====================================== other functions
    @abc.abstractmethod
    def xfn(self, data):
        """
        Parameters
        ----------
        data : np.ndarray
            shape = (batch, time_series, features)
        """
        return data

    @abc.abstractmethod
    def yfn(self, data):
        """
        Parameters
        ----------
        data : np.ndarray
            shape = (batch, time_series, features)
        """
        return data
