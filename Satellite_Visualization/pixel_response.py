import glob
from os import path

import numpy as np
import rasterio as rio

from . import settings, utils

__all__ = ["PixelResponseBuilder"]


class NonBinaryResponseError(Exception):
    pass


class PixelResponseBuilder:

    # It is really not necessary to use a class for this, but we do so for the sake of
    # API consistency with the `pixel_features` module
    def __init__(self, *, tree_val=None, nontree_val=None):

        if tree_val is None:
            tree_val = settings.RESPONSE_TREE_VAL
        self.tree_val = tree_val

        if nontree_val is None:
            nontree_val = settings.RESPONSE_NONTREE_VAL
        self.nontree_val = nontree_val

    def build_response_from_arr(self, img_binary):

        response_arr = img_binary.copy()
        response_arr[response_arr == self.tree_val] = 1
        response_arr[response_arr == self.nontree_val] = 0

        # check that the provided `img_binary` is actually binary, i.e., consists only
        # of `tree_val` and `nontree_val` values
        if ((response_arr != 0) & (response_arr != 1)).any():
            raise NonBinaryResponseError

        return response_arr.flatten()

    def build_response_from_filepath(self, img_filepath):

        with rio.open(img_filepath) as src:
            img_binary = src.read(1)

        try:
            return self.build_response_from_arr(img_binary)
        except NonBinaryResponseError:
            raise ValueError(
                f"The response mask {img_filepath} must consist of only {self.tree_val}"
                f" (tree) and {self.nontree_val} (non-tree) pixel values"
            )

    def build_response(
        self,
        *,
        split_df=None,
        response_img_dir=None,
        response_img_filepaths=None,
        img_filename_pattern=None,
        method=None,
        img_cluster=None,
    ):

        if split_df is not None:
            if response_img_dir is None:
                raise ValueError(
                    "If `split_df` is provided, `response_img_dir` must also be"
                    " provided"
                )
            if method is None:
                if "img_cluster" in split_df:
                    method = "cluster-II"
                else:
                    method = "cluster-I"

            if method == "cluster-I":
                img_filepaths = split_df[split_df["train"]]["img_filepath"]
            else:
                if img_cluster is None:
                    raise ValueError(
                        "If `method` is 'cluster-II', `img_cluster` must be provided"
                    )
                img_filepaths = utils.get_img_filepaths(split_df, img_cluster, True)

            response_img_filepaths = img_filepaths.apply(
                lambda filepath: path.join(response_img_dir, path.basename(filepath))
            )
        else:
            if response_img_filepaths is None:
                if img_filename_pattern is None:
                    img_filename_pattern = settings.IMG_FILENAME_PATTERN
                if response_img_dir is None:
                    raise ValueError(
                        "Either `split_df`, `response_img_filepaths` or "
                        "`response_img_dir` must be provided"
                    )

                response_img_filepaths = glob.glob(
                    path.join(response_img_dir, img_filename_pattern)
                )
            # TODO: `response_img_filepaths`

        # no need for dask here
        values = []
        for response_img_filepath in response_img_filepaths:
            values.append(self.build_response_from_filepath(response_img_filepath))

        return np.vstack(values).flatten()