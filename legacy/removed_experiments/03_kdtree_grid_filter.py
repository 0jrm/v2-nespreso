"""ARCHIVED — monolith __main__ block (Phase 9 removal, commit 2cbb2b1).

Source: legacy/monolith/singleFileModel_SAT_stats4verticalProj_meeting20260203.py
lines ~253-293. KD-tree grid spatial coverage plot.

NOT wired. Code below is preserved verbatim (still commented).
"""

    # # %%
    # from scipy.spatial import cKDTree

    # # lon_min = -88
    # # lon_max = -82
    # lon_min = np.floor(np.min(full_dataset.LON))
    # lon_max =  np.ceil(np.max(full_dataset.LON))
    # lat_min = np.floor(np.min(full_dataset.LAT))
    # lat_max =  np.ceil(np.max(full_dataset.LAT))

    # # Define grid spacing
    # grid_spacing = 0.1  # degrees

    # # Create the grid within the bounding box
    # lats_grid = np.arange(lat_min, lat_max + grid_spacing, grid_spacing)
    # lons_grid = np.arange(lon_min, lon_max + grid_spacing, grid_spacing)

    # # Use meshgrid to create a grid of coordinates
    # lats_mesh, lons_mesh = np.meshgrid(lats_grid, lons_grid)

    # # Flatten the meshgrid arrays to obtain the full list of coordinates
    # grid_points = np.vstack([lats_mesh.ravel(), lons_mesh.ravel()]).T

    # # Build a KD-Tree with the original LAT and LON data
    # data_points = np.vstack([full_dataset.LAT, full_dataset.LON]).T
    # tree = cKDTree(data_points)

    # # Query the tree for each grid point to find the distance to the nearest data point
    # distances, _ = tree.query(grid_points, distance_upper_bound=0.5)

    # # Filter the grid points where the distance is infinity (no points within 0.2 degrees)
    # filtered_grid_points = grid_points[distances != np.inf]

    # # Plot original data points and the filtered grid points
    # plt.figure(figsize=(10, 8))
    # plt.scatter(filtered_grid_points[:, 1], filtered_grid_points[:, 0], color='red', label='Filtered Grid Points', s=0.2)
    # plt.xlabel('Longitude')
    # plt.ylabel('Latitude')
    # plt.title('Points within 0.5 degrees of original data')
    # plt.legend()
    # plt.grid(True)
