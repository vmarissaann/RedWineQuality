# Source: CSMODEL T2 AY22-23
# Notebook 9 - Collaborative Filtering on Movie Dataset

import numpy as np
import pandas as pd


class IICollabFiltering(object):

    def __init__(self, k=3):
        """Class constructor for KMeans
        Arguments:
            k {int} -- number of similar items to consider. Default is 3.
        """
        self.k = k

    def get_row_mean(self, data):
        """Returns the mean of each row in the DataFrame or the mean of the
        Series. If the parameter data is a DataFrame, the function will
        return a Series containing the mean of each row in the DataFrame. If
        the parameter data is a Series, the function will return a np.float64
        which is the mean of the Series. This function should not consider
        blank ratings represented as NaN.

        Arguments:
            data {DataFrame or Series} -- dataset
        Returns:
            Series or np.float64 -- row mean
        """

        if len(data.shape) == 1:
            axis = 0
        else:
            axis = 1

        return data.mean(axis=axis)

    def normalize_data(self, data, row_mean):
        """Returns the data normalized by subtracting the row mean.

        For the arguments point1 and point2, you can only pass these
        combinations of data types:
        - DataFrame and Series -- returns DataFrame
        - Series and np.float64 -- returns Series

        For a DataFrame and a Series, if the shape of the DataFrame is
        (3, 2), the shape of the Series should be (3,) to enable broadcasting.
        This operation will result to a DataFrame of shape (3, 2)

        Arguments:
            data {DataFrame or Series} -- dataset
            row_mean {Series or np.float64} -- mean of each row
        Returns:
            DataFrame or Series -- normalized data
        """

        # Check if the combination of parameters is correct
        # Normalize the parameter data by parameter row_mean.
        return data.subtract(row_mean, axis=0)

    def get_cosine_similarity(self, vector1, vector2):
        """Returns the cosine similarity between two vectors. These vectors can
        be represented as 2 Series objects. This function can also compute the
        cosine similarity between a list of vectors (represented as a
        DataFrame) and a single vector (represented as a Series), using
        broadcasting.

        For the arguments vector1 and vector2, you can only pass these
        combinations of data types:
        - Series and Series -- returns np.float64
        - DataFrame and Series -- returns pd.Series

        For a DataFrame and a Series, if the shape of the DataFrame is
        (3, 2), the shape of the Series should be (2,) to enable broadcasting.
        This operation will result to a Series of shape (3,)

        Arguments:
            vector1 {Series or DataFrame} - vector
            vector2 {Series or DataFrame} - vector
        Returns:
            np.float64 or pd.Series -- contains the cosine similarity between
            two vectors
        """

        # Check if the parameter data is a Series or a DataFrame
        # for numerator
        num_axis = 0 if len(vector1.shape) == 1 and len(vector2.shape) == 1 else 1
        
        # for sqrt(A^2)
        axis_A = 0 if len(vector1.shape) == 1 else  1
        
        # for sqrt(B^2)
        axis_B = 0 if len(vector2.shape) == 1 else 1

        # Compute the cosine similarity between the two parameters.
        sop = (vector1 * vector2).sum(axis=num_axis)
        vector1_sqA2 = np.sqrt((vector1 ** 2).sum(axis=axis_A))
        vector2_sqB2 = np.sqrt((vector2 ** 2).sum(axis=axis_B))
        return sop / (vector1_sqA2 * vector2_sqB2)

    def get_k_similar(self, data, vector):
        """Returns two values - the indices of the top k similar items to the
        vector from the dataset, and a Series representing their similarity
        values to the vector. We find the top k items from the data which
        are highly similar to the vector.

        Arguments:
            data {DataFrame} -- dataset
            vector {Series} -- vector
        Returns:
            Index -- indices of the top k similar items to the vector
            Series -- computed similarity of the top k similar items to the
            vector
        """

        # Normalize parameters data and vector
        data_norm = self.normalize_data(data, self.get_row_mean(data))
        vector_norm = self.normalize_data(vector, self.get_row_mean(vector))

        # Get the cosine similarity between the normalized data and
        # vector
        cos_sim = self.get_cosine_similarity(data_norm, vector_norm)

        # Get the INDICES of the top k most similar items based on
        # the cosine similarity values
        k_sim = cos_sim.nlargest(self.k)

        # Return indeces and similarity values
        return k_sim.index, k_sim

    def get_rating(self, data, new_wine):
        """Returns the extrapolated rating for the item in row index from the
        user in column column based on similar items.

        The algorithm for this function is as follows:
        1. Get k similar items.
        2. Compute for the rating using the similarity values and the raw
        ratings for the k similar items.

        Arguments:
            data {DataFrame} -- dataset
            index {int} -- row of the item
            column {int} -- column of the user
        Returns:
            np.float64 -- extrapolated rating based on similar ratings
        """

        # before_df = data.iloc[:index, :]
        # after_df = data.iloc[index + 1:, :]
        # new_data = pd.concat([before_df, after_df])
        # vector = data.iloc[index, :]

        # Get top k items that are similar to the parameter vector
        k_sim_inds, k_sim_vals = self.get_k_similar(data, new_wine)
        k_ratings = data.iloc[k_sim_inds, :].loc[:, "quality"]

        # Compute for the rating using the similarity values and the raw
        # ratings for the k similar items.
        rating = round((k_sim_vals * k_ratings).sum() / k_sim_vals.sum())

        return rating
