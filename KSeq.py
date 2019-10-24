import numpy as np
import pandas as pd
from keras import utils


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
                 x_fn=None,
                 y_fn=None,
                 verbosity=False):
        """
        Parameters
        ----------
        dataset : pd.DataFrame
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
        x_fn , y_fn: function | None
            np.ndarray -> any
        verbosity = bool
        """

        if verbosity:
            print('TS generator initializing...')
        # ...................... preparing args
        self.dataset = dataset.copy()
        self.input_length, self.output_length = input_length, output_length
        self.feature_col, self.target_col = feature_columns, target_columns
        self.xfn, self.yfn = x_fn, y_fn
        # ......................
        if sort != 0:
            sort = {1: True, -1: False}[sort]
            self.dataset.sort_index(ascending=sort, inplace=True)
            if verbosity:
                print('sorting : ascending', sort)
        else:
            if verbosity:
                print("sorting : no")
        # ......................
        self.X = self.dataset[self.feature_col].values
        self.Y = self.dataset[self.target_col].values
        # ......................
        self.sample_n = int(len(self.dataset) - (self.input_length + self.output_length) + 1)
        self.batch_size = batch_size
        x_ts = [np.arange(i, i + input_length) for i in range(self.sample_n)]
        y_ts = [np.arange(i + input_length, i + input_length + output_length) for i in range(self.sample_n)]
        self.x_ts = np.array(x_ts, dtype=np.uint32)
        self.y_ts = np.array(y_ts, dtype=np.uint32)
        # ......................
        self.meta_index = np.arange(self.sample_n, dtype=np.uint32)
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
            raise Exception(e)

    # -------------- abs mode
    def __getitem__abs__(self, idx):
        x = self.X[self.x_ts[idx, :], :].copy()
        if len(x) == 0:
            x = None
        else:
            if x.ndim == 2:
                x = x[np.newaxis, :, :]
            if self.xfn is not None:
                x = self.xfn(x)

        y = self.Y[self.y_ts[idx, :], :].copy()
        if len(y) == 0:
            y = None
        else:
            if y.ndim == 2:
                y = y[np.newaxis, :, :]
            if self.yfn is not None:
                y = self.yfn(y)

        return x, y

    # -------------- batch mode
    def __getitem__batch__(self, idx):  # needs more works!
        assert idx.__class__ == int, 'in batch mode, index has to be a single int'
        assert 0 <= idx < self.batch_n, "batch {} doesn't exist".format(idx)
        idx = self.meta_index[idx * self.batch_size:(idx + 1) * self.batch_size]
        return self.__getitem__abs__(idx)

    # ======================================
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

    # ====================================== batch_size
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
