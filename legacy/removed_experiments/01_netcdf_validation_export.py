"""ARCHIVED — monolith __main__ block (Phase 9 removal, commit 2cbb2b1).

Source: legacy/monolith/singleFileModel_SAT_stats4verticalProj_meeting20260203.py
lines ~166-222. NetCDF validation export (Test_dataset.nc).

NOT wired. Code below is preserved verbatim (still commented).
"""

        # #create a netcdf file with the validation dataset
    # sst_val = get_inputs(val_loader, device)[:,-2]
    # lat_val = val_dataset.dataset.LAT[val_indices]
    # lon_val = val_dataset.dataset.LON[val_indices]
    # date_val = datenums_to_datetimes(val_dataset.dataset.TIME[val_indices])
    # T_profiles_val = original_profiles[:,0,:]
    # S_profiles_val = original_profiles[:,1,:]
    # depth = np.arange(0,1801)

    # sst_val.shape, lat_val.shape, lon_val.shape, type(date_val), type(date_val[0]), len(date_val), T_profiles_val.shape, S_profiles_val.shape, depth.shape

    # from netCDF4 import Dataset
    # # import numpy as np
    # from datetime import datetime, timedelta

    # def create_netcdf(filename, sst_val, lat_val, lon_val, date_val, T_profiles_val, S_profiles_val, depth):
    #     with Dataset(filename, 'w', format='NETCDF4') as nc:
    #         # Dimensions
    #         nc.createDimension('time', len(date_val))
    #         nc.createDimension('lat', len(lat_val))
    #         nc.createDimension('lon', len(lon_val))
    #         nc.createDimension('depth', len(depth))

    #         # Variables
    #         times = nc.createVariable('time', 'f8', ('time',))
    #         lats = nc.createVariable('lat', 'f4', ('lat',))
    #         lons = nc.createVariable('lon', 'f4', ('lon',))
    #         depths = nc.createVariable('depth', 'f4', ('depth',))
    #         sst = nc.createVariable('sst', 'f4', ('time',))
    #         T_profiles = nc.createVariable('T_profiles', 'f4', ('depth', 'time'))
    #         S_profiles = nc.createVariable('S_profiles', 'f4', ('depth', 'time'))

    #         # Convert datetime to numeric time values
    #         ref_date = datetime(1900, 1, 1)
    #         numeric_dates = [(d - ref_date).total_seconds() for d in date_val]
    #         times[:] = numeric_dates

    #         # Assign data
    #         lats[:] = lat_val
    #         lons[:] = lon_val
    #         depths[:] = depth
    #         sst[:] = sst_val
    #         T_profiles[:, :] = T_profiles_val
    #         S_profiles[:, :] = S_profiles_val

    #         # Add attributes
    #         nc.description = 'Dataset used for acquiring statistics for ISOP, GEM and NeSPReSO methods. Contains SST, latitude, longitude, date, temperature profiles, salinity profiles, and depth.'
    #         sst.units = 'Celsius'
    #         lats.units = 'degrees'
    #         lons.units = 'degrees'
    #         depths.units = 'meter'
    #         times.units = 'seconds since 1900-01-01 00:00:00'
    #         T_profiles.units = 'Celsius'
    #         S_profiles.units = 'PSU'

    #         print(f"NetCDF file '{filename}' created successfully.")
