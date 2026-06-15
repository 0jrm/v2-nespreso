"""Temperature/salinity profile dataset for NeSPReSO."""

import mat73
import numpy as np
import torch
from sklearn.decomposition import PCA

from nespreso.data.gem import calc_gem as _calc_gem
from nespreso.data.gem import get_gem_profiles as _get_gem_profiles
from nespreso.data.pca import sklearn_inverse_transform_pcs
from nespreso.io.argo import load_argo_mat
from nespreso.io.satellite import load_satellite_data_for_dataset
from nespreso.utils.time import datenum_to_datetime

debug = False


def _resolve_debug():
    import sys

    main = sys.modules.get("__main__")
    if main is not None and hasattr(main, "debug"):
        return bool(main.debug)
    return debug


class TemperatureSalinityDataset(torch.utils.data.Dataset):
    """
    Custom dataset for temperature and salinity profiles.

    Attributes:
    - TEMP: Temperature profiles matrix.
    - SAL: Salinity profiles matrix.
    - SSH: Sea Surface Height vector.
    - pca_temp: PCA model for temperature profiles.
    - pca_sal: PCA model for salinity profiles.
    - temp_pcs: Transformed temperature profiles using PCA.
    - sal_pcs: Transformed salinity profiles using PCA.
    """

    def __init__(
        self,
        n_components=15,
        input_params=None,
        max_depth=2000,
        min_depth=20,
        data_path=None,
        aviso_folder=None,
        sst_folder=None,
        sss_folder=None,
        min_lat=18.0,
        max_lat=31.0,
        min_lon=-98.0,
        max_lon=-81.0,
        ex_lat=23.0,
        ex_lon=-90.0,
    ):
        """

        Args:
        - path (str): File path to the dataset.
        - n_components (int): Number of PCA components to retain.
        """
        self.n_components = n_components
        self.data_path = data_path or "/unity/g2/jmiranda/SubsurfaceFields/Data/ARGO_GoM_20220920.mat"
        self.aviso_folder = aviso_folder or "/unity/f1/ozavala/DATA/GOFFISH/AVISO/GoM/"
        self.sst_folder = sst_folder or "/unity/f1/ozavala/DATA/GOFFISH/SST/OISST"
        self.sss_folder = sss_folder or "/Net/work/ozavala/DATA/GOFFISH/SSS/SMAP_Global/"

        self.max_depth = max_depth
        self.min_depth = min_depth  # data quality is poor above 20m

        self.data, self.TIME, self.LAT, self.LON, self.SH1950 = load_argo_mat(self.data_path)
        self.min_lat = min_lat
        self.max_lat = max_lat
        self.min_lon = min_lon
        self.max_lon = max_lon
        self.ex_lat = ex_lat
        self.ex_lon = ex_lon

        self.input_params = input_params

        self.SSS, self.SST, self.AVISO_ADT = self._load_satellite_data()
        self.satSSS, self.satSST, self.sat_ADT = np.copy(self.SSS), np.copy(self.SST), np.copy(self.AVISO_ADT)  # backup

        # self.adjust_ADT()

        self.valid_mask = self._get_valid_mask(self.data)
        valid_mask = self.valid_mask
        (
            self.TEMP,
            self.SAL,
            self.AVISO_ADT,
            self.SST,
            self.SSS,
            self.TIME,
            self.LAT,
            self.LON,
            self.SH1950,
            self.PRES,
        ) = self._filter_and_fill_data(self.data, valid_mask)

        # Applying PCA
        self.temp_pcs, self.pca_temp = self._apply_pca(self.TEMP, self.n_components)
        self.sal_pcs, self.pca_sal = self._apply_pca(self.SAL, self.n_components)

    # def adjust_ADT(self):
    #     #  Remove the daily mean ADT from the AVISO_ADT

    #     gom_mean = xr.load_dataset('/unity/g2/jmiranda/SubsurfaceFields/Data/gom_mean_adt_2013_2022.nc')
    #     self.mean_adt = gom_mean.gom_mean_adt.values
    #     self.time_mean_adt = np.floor(gom_mean.time.values)

    #     for i, t in enumerate(self.data['TIME']):
    #         t = np.floor(t)
    #         i_match = self.time_mean_adt == t
    #         if np.sum(i_match) == 0:
    #             print(f"No match: idx: {i}\t .nc time: {t}")
    #         elif np.sum(i_match) > 1:
    #             print(f"Multiple matches at dx: {i}\t .nc time: {t}")
    #         # else:
    #         #     print(f"Single match: {i} \t {t} \t {mean_adt[i_match]} ")

    #         self.AVISO_ADT[i] = self.AVISO_ADT[i] - self.mean_adt[i_match]

    def reload(self):
        # in case we want to change parameters...
        self.SSS, self.SST, self.AVISO_ADT = np.copy(self.satSSS), np.copy(self.satSST), np.copy(self.sat_ADT)
        self.data = mat73.loadmat(self.data_path)
        # self.adjust_ADT()

        valid_mask = self._get_valid_mask(self.data)
        (
            self.TEMP,
            self.SAL,
            self.AVISO_ADT,
            self.SST,
            self.SSS,
            self.TIME,
            self.LAT,
            self.LON,
            self.SH1950,
            self.PRES,
        ) = self._filter_and_fill_data(self.data, valid_mask)

        # Applying PCA
        self.temp_pcs, self.pca_temp = self._apply_pca(self.TEMP, self.n_components)
        self.sal_pcs, self.pca_sal = self._apply_pca(self.SAL, self.n_components)

    def _load_satellite_data(self):
        return load_satellite_data_for_dataset(
            self.TIME,
            self.LAT,
            self.LON,
            self.aviso_folder,
            self.sst_folder,
            self.sss_folder,
            self.min_lat,
            self.max_lat,
            self.min_lon,
            self.max_lon,
            self.ex_lat,
            self.ex_lon,
            debug=_resolve_debug(),
        )

    def __getitem__(self, idx):
        """
        Args:
        - idx (int): Index of the profile.

        Returns:
        - tuple: input values and concatenated PCA components for temperature and salinity.
        """

        inputs = []

        if self.input_params["timecos"]:
            inputs.append(np.cos(2 * np.pi * (self.TIME[idx] % 365) / 365))

        if self.input_params["timesin"]:
            inputs.append(np.sin(2 * np.pi * (self.TIME[idx] % 365) / 365))

        if self.input_params["latcos"]:
            inputs.append(np.cos(2 * np.pi * (self.LAT[idx] / 180)))

        if self.input_params["latsin"]:
            inputs.append(np.sin(2 * np.pi * (self.LAT[idx] / 180)))

        if self.input_params["loncos"]:
            inputs.append(np.cos(2 * np.pi * (self.LON[idx] / 360)))

        if self.input_params["lonsin"]:
            inputs.append(np.sin(2 * np.pi * (self.LON[idx] / 360)))

        if self.input_params["sat"]:
            if self.input_params["sss"]:
                # inputs.append(self.SAL[0, idx])
                inputs.append(self.SSS[idx])

            if self.input_params["sst"]:
                inputs.append(self.SST[idx] - 273.15)  # convert from Kelvin to Celsius

            if self.input_params["ssh"]:
                # inputs.append(self.SH1950[idx]) #Uses profile SSH
                inputs.append(self.AVISO_ADT[idx])  # Uses satellite SSH
        else:
            if self.input_params["sss"]:
                inputs.append(self.SAL[0, idx])
                # inputs.append(self.SSS[idx])

            if self.input_params["sst"]:
                inputs.append(self.TEMP[0, idx])  # First value of temperature profile
                # inputs.append(self.SST[idx])

            if self.input_params["ssh"]:
                inputs.append(self.SH1950[idx])  # Uses profile SSH
                # inputs.append(self.AVISO_ADT[idx]) #Uses satellite SSH

        inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
        profiles = torch.tensor(np.hstack([self.temp_pcs[:, idx], self.sal_pcs[:, idx]]), dtype=torch.float32)
        return inputs_tensor, profiles

    def _get_valid_mask(self, data):
        """Internal method to get mask of valid profiles based on missing values."""

        ssh_mask = ~np.isnan(self.AVISO_ADT)
        sst_mask = ~np.isnan(self.SST)
        sss_mask = ~np.isnan(self.SSS)

        combined_mask = np.logical_and(sst_mask, ssh_mask)
        print(f"Filtered dataset (sal/temp/ssh/sst) contains {np.sum(combined_mask)} profiles.")
        combined_mask = np.logical_and(combined_mask, sss_mask)
        print(f"Filtered dataset (sal/temp/ssh/sst/sss) contains {np.sum(combined_mask)} profiles.")

        return combined_mask

    def _filter_and_fill_data(self, data, valid_mask):
        """Internal method to filter data using the mask and fill missing values."""
        TEMP = data["TEMP"][self.min_depth : self.max_depth + 1, valid_mask]
        SAL = data["SAL"][self.min_depth : self.max_depth + 1, valid_mask]
        PRES = data["PRES"][self.min_depth : self.max_depth + 1, valid_mask]
        LAT = data["LAT"][valid_mask]
        LON = data["LON"][valid_mask]
        ADT = data["SH1950"][valid_mask]
        TIME = data["TIME"][valid_mask]

        SSH = self.AVISO_ADT[valid_mask]
        SST = self.SST[valid_mask]
        SSS = self.SSS[valid_mask]

        # Fill missing values using interpolation
        for i in range(TEMP.shape[1]):
            # let's fill nans at the beginning and end of the profiles
            first_nan = np.where(~np.isnan(TEMP[:, i]))[0][0]
            last_nan = np.where(~np.isnan(TEMP[:, i]))[0][-1]
            if np.isnan(TEMP[0, i]):
                TEMP[:first_nan, i] = TEMP[first_nan, i]
            if np.isnan(TEMP[-1, i]):
                TEMP[last_nan:, i] = TEMP[last_nan, i]

            valid_temp_idx = np.where(~np.isnan(TEMP[:, i]))[0]
            TEMP[:, i] = np.interp(range(TEMP.shape[0]), valid_temp_idx, TEMP[valid_temp_idx, i])
            valid_sal_idx = np.where(~np.isnan(SAL[:, i]))[0]
            SAL[:, i] = np.interp(range(SAL.shape[0]), valid_sal_idx, SAL[valid_sal_idx, i])

        return TEMP, SAL, SSH, SST, SSS, TIME, LAT, LON, ADT, PRES

    def _apply_pca(self, data, n_components):
        """Internal method to apply PCA transformation to the data."""
        pca = PCA(n_components=n_components)
        pcs = pca.fit_transform(data.T).T
        return pcs, pca

    def __len__(self):
        """Returns number of profiles in the dataset."""
        return self.TEMP.shape[1]

    def inverse_transform(self, pcs):
        """
        Inverse the PCA transformation.

        Args:
        - pcs (numpy.ndarray): Concatenated PCA components for temperature and salinity.

        Returns:
        - tuple: Inversed temperature and salinity profiles.
        """
        return sklearn_inverse_transform_pcs(pcs, self.pca_temp, self.pca_sal, self.n_components)

    def get_profiles(self, indices, pca_approx=False):
        """
        Returns temperature and salinity profiles for the given indices.

        Args:
        - indices (list or numpy.ndarray): List of indices for which profiles are needed.
        - pca_approx (bool): Flag to return PCA approximated profiles if True,
                             or original profiles if False.

        Returns:
        - numpy.ndarray: concatenated temperature and salinity profiles in the required format for visualization.
        """
        indices = np.atleast_1d(indices)
        if pca_approx:
            # Concatenate temp and sal PCA components for the given indices
            concatenated_pcs = np.hstack([self.temp_pcs[:, indices].T, self.sal_pcs[:, indices].T])
            # Obtain PCA approximations using the concatenated components
            temp_profiles, sal_profiles = self.inverse_transform(concatenated_pcs)
        else:
            temp_profiles = self.TEMP[:, indices]
            sal_profiles = self.SAL[:, indices]

        # Stack along the third dimension
        profiles_array = np.stack([temp_profiles, sal_profiles], axis=1)

        return profiles_array

    def calc_gem(self, ignore_indices, degree=7, sat_ssh=False):
        """
        Calculates this dataset's polyfits for the GEM profiles for each month.

        Args:
        - degree: Degree of the polynomial fit. Default is 7.
        - sat_ssh: Flag to use satellite SSH instead of profile SSH. Uses measured SSH as default.

        Returns:
        - nothing, but saves the polyfits in the attributes `self.gem_T_polyfits` and `self.gem_S_polyfits` for each month.
        """
        return _calc_gem(self, ignore_indices, degree=degree, sat_ssh=sat_ssh)

    def get_gem_profiles(self, indices, sat_ssh=True):
        """
        Generates the GEM profiles for the given indices, month by month.

        Args:
        - indices (list or numpy.ndarray): Indices for which profiles are needed.
        - sat_ssh (bool): Flag to use satellite SSH instead of profile SSH. Uses measured SSH as default.

        Returns:
        - numpy.ndarray: concatenated temperature and salinity profiles in the required format for visualization.
        """
        return _get_gem_profiles(self, indices, sat_ssh=sat_ssh)

    def get_inputs(self, idx):
        sst_inputs = self.TEMP[0, idx]
        ssh_inputs = self.AVISO_ADT[idx]
        return sst_inputs, ssh_inputs

    def get_lat_lon_date(self, idx):
        lat = self.LAT[idx]
        lon = self.LON[idx]
        date = self.TIME[idx]
        return lat, lon, date

    def get_pca_weights(self):
        """
        Get the concatenated vector of the variance represented by each PC of the temperature and salinity datasets,
        divided by the total variance of each PCS.

        Returns:
        - numpy.ndarray: Concatenated vector of variances for temperature and salinity PCs.
        """
        temp_variance = self.pca_temp.explained_variance_ratio_ / self.temp_pcs.var(axis=1)
        sal_variance = self.pca_sal.explained_variance_ratio_ / self.sal_pcs.var(axis=1)
        concatenated_variance = np.concatenate([temp_variance, sal_variance])
        return concatenated_variance

