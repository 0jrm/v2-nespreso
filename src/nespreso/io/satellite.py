"""Satellite data loading and interpolation for NeSPReSO."""

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm

from nespreso.io.satellite_readers import get_aviso_by_date, get_sss_by_date, get_sst_by_date


def load_satellite_data(TIME, LAT, LON):
    """
    New method to load SST and SSH data
    """
    aviso_folder = "/unity/f1/ozavala/DATA/GOFFISH/AVISO/GoM/"
    sst_folder = "/unity/f1/ozavala/DATA/GOFFISH/SST/OISST"
    sss_folder = "/Net/work/ozavala/DATA/GOFFISH/SSS/SMAP_Global/"
    min_lat = 18.0
    max_lat = 31.0
    min_lon = -98.0
    max_lon = -81.0
    ex_lon = -88.0
    ex_lat = 23.0
    unique_dates = sorted(list(set(TIME)))
    sss_data = np.nan * np.ones(len(TIME))
    sst_data = np.nan * np.ones(len(TIME))
    aviso_data = np.nan * np.ones(len(TIME))
    bbox = (min_lat, max_lat, min_lon, max_lon)

    # # Convert serialized date numbers to date objects
    # base_date = datetime(1, 1, 1)

    # for idx, serialized_date in enumerate(unique_dates):
    for idx, c_date in enumerate(unique_dates):
        print(f"Querying satellite data:  {c_date.date()}.")
        # c_date = base_date + timedelta(days=float(serialized_date))
        date_idx = np.array(
            [date_obj == c_date for date_obj in TIME]
        )  # Ensure both sides of the comparison are datetime objects
        coordinates = np.array([LAT[date_idx], LON[date_idx]]).T

        # TODO: get errors

        try:
            sss_datapoint, lats, lons = get_sss_by_date(sss_folder, c_date, bbox)
            interpolator = RegularGridInterpolator(
                (lats, lons), sss_datapoint.sss_smap_40km.values, bounds_error=False, fill_value=None
            )
            sss_data[date_idx] = interpolator(coordinates)
            if (sss_data[date_idx] < 0).any() or (sss_data[date_idx] > 45).any():
                sss_data[date_idx] = np.nan
                print(
                    f"Invalid SSS on date {c_date}, value: {sss_data[date_idx]}, coordinates: {coordinates}, lats: {lats}, lons: {lons}"
                )
        except Exception as e:
            print(f"SSS not found on date {c_date}. Error: {e}")

        try:
            aviso_adt, aviso_lats, aviso_lons = get_aviso_by_date(aviso_folder, c_date, bbox)
            # Generate 2D arrays of latitudes and longitudes for each point in the grid
            lons, lats = np.meshgrid(aviso_lons, aviso_lats)

            # Create the mask based on inclusion criteria (bbox) and exclusion criteria
            inclusion_mask = (lats >= min_lat) & (lats <= max_lat) & (lons >= min_lon) & (lons <= max_lon)
            exclusion_mask = (lats < ex_lat) & (lons > ex_lon)

            # Combine masks to exclude the specified area
            combined_mask = inclusion_mask & ~exclusion_mask

            # Apply the combined mask to filter the data before calculating the mean
            daily_avg = np.nanmean(aviso_adt.adt.values[combined_mask])

            interpolator_ssh = RegularGridInterpolator(
                (aviso_lats, aviso_lons), aviso_adt.adt.values, bounds_error=False, fill_value=None
            )
            aviso_data[date_idx] = interpolator_ssh(coordinates) - daily_avg

        except Exception as e:
            print("AVISO not found for date ", c_date, "Error: ", str(e))
            continue

        try:
            sst_date, sst_lats, sst_lons = get_sst_by_date(sst_folder, c_date, bbox)
            interpolator_sst = RegularGridInterpolator(
                (sst_lats, sst_lons), sst_date.analysed_sst.values[0], bounds_error=False, fill_value=None
            )
            sst_data[date_idx] = interpolator_sst(coordinates)
            if (sst_data[date_idx] < 0).any() or (sst_data[date_idx] > 350).any():
                sst_data[date_idx] = np.nan
                print(
                    f"Invalid SST on date {c_date}, value: {sst_data[date_idx]}, coordinates: {coordinates}, lats: {sst_lats}, lons: {sst_lons}"
                )
        except Exception as e:
            print("SST not found for date ", c_date, "Error: ", str(e))
            continue

        # Check if data was actually filled
        if np.isnan(aviso_data[date_idx]).all():
            print(f"No AVISO data for date {c_date}")
        if np.isnan(sst_data[date_idx]).all():
            print(f"No SST data for date {c_date}")

    return sss_data, sst_data, aviso_data


def load_satellite_data_for_dataset(
    TIME,
    LAT,
    LON,
    aviso_folder,
    sst_folder,
    sss_folder,
    min_lat,
    max_lat,
    min_lon,
    max_lon,
    ex_lat,
    ex_lon,
    debug=False,
):
    """
    Load Sea Surface Temperature (SST), Sea Surface Salinity (SSS), and Sea Surface Height (SSH) data.

    This method loads and interpolates satellite data for SST, SSS, and SSH within a specified geographic bounding box
    and time range. It also includes an optional debugging mode that logs data loading failures.

    Returns:
        tuple: Tuple containing arrays for SSS, SST, and AVISO data.
    """
    bbox = (min_lat, max_lat, min_lon, max_lon)
    sss_data, sst_data, aviso_data = (
        np.nan * np.ones(len(TIME)),
        np.nan * np.ones(len(TIME)),
        np.nan * np.ones(len(TIME)),
    )
    error_log = []

    for idx, c_date in tqdm(
        enumerate(sorted(set(TIME))), total=len(set(TIME)), desc="Loading Satellite Data"
    ):
        date_idx = np.array([date_obj == c_date for date_obj in TIME])
        coordinates = np.array([LAT[date_idx], LON[date_idx]]).T

        # SSS data loading
        try:
            sss_datapoint, lats, lons = get_sss_by_date(sss_folder, c_date, bbox)
            interpolator = RegularGridInterpolator(
                (lats, lons), sss_datapoint.sss_smap_40km.values, bounds_error=False, fill_value=None
            )
            sss_data[date_idx] = interpolator(coordinates)
            if (sss_data[date_idx] < 0).any() or (sss_data[date_idx] > 45).any():
                sss_data[date_idx] = np.nan
                if debug:
                    error_log.append(
                        {
                            "Date": c_date,
                            "Parameter": "SSS",
                            "Filename": sss_datapoint.filename,
                            "Reason": "Invalid SSS values",
                        }
                    )
        except Exception as e:
            if debug:
                error_log.append({"Date": c_date, "Parameter": "SSS", "Filename": None, "Reason": str(e)})

        # AVISO data loading
        try:
            aviso_adt, aviso_lats, aviso_lons = get_aviso_by_date(aviso_folder, c_date, bbox)
            lons, lats = np.meshgrid(aviso_lons, aviso_lats)
            inclusion_mask = (lats >= min_lat) & (lats <= max_lat) & (lons >= min_lon) & (lons <= max_lon)
            exclusion_mask = (lats < ex_lat) & (lons > ex_lon)
            combined_mask = inclusion_mask & ~exclusion_mask
            daily_avg = np.nanmean(aviso_adt.adt.values[combined_mask])
            interpolator_ssh = RegularGridInterpolator(
                (aviso_lats, aviso_lons), aviso_adt.adt.values, bounds_error=False, fill_value=None
            )
            aviso_data[date_idx] = interpolator_ssh(coordinates) - daily_avg
        except Exception as e:
            if debug:
                error_log.append({"Date": c_date, "Parameter": "AVISO", "Filename": None, "Reason": str(e)})

        # SST data loading
        try:
            sst_date, sst_lats, sst_lons = get_sst_by_date(sst_folder, c_date, bbox)
            interpolator_sst = RegularGridInterpolator(
                (sst_lats, sst_lons), sst_date.analysed_sst.values[0], bounds_error=False, fill_value=None
            )
            sst_data[date_idx] = interpolator_sst(coordinates)
        except Exception as e:
            if debug:
                error_log.append({"Date": c_date, "Parameter": "SST", "Filename": None, "Reason": str(e)})

    if debug and error_log:
        # You can choose to save this as a CSV or any other format
        print("Error log:", error_log)

    return sss_data, sst_data, aviso_data
