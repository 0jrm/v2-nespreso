"""ARCHIVED — monolith __main__ block (Phase 9 removal, commit 2cbb2b1).

Source: legacy/monolith/singleFileModel_SAT_stats4verticalProj_meeting20260203.py
lines ~376-466. Nature-run T-S by SSH bin.

NOT wired. Code below is preserved verbatim (still commented).
"""

    # # Compare T/S diagrams for ssh ranges
    # ssh_nature_run = aggregate_from_mat(folder_path, 'ssh10')['ssh10']
    # T_nature_run = aggregate_from_mat(folder_path, 'temp10')['temp10']
    # S_nature_run = aggregate_from_mat(folder_path, 'sal10')['sal10']

    # import matplotlib.colors as mcolors

    # def plot_ts_profiles(datasets, dataset_labels, sigma_theta, Sg, Tg, cores, cmap_name='viridis'):
    #     """
    #     Plots T-S profiles from multiple datasets on the same plot.

    #     Parameters:
    #     - datasets: List of tuples [(TEMP1, SAL1), (TEMP2, SAL2), ...]
    #                 Each tuple contains temperature and salinity data.
    #     - dataset_labels: List of labels corresponding to each dataset.
    #     - sigma_theta: 2D array of sigma_theta values for contour plotting.
    #     - Sg: 2D array of salinity grid values for contour plotting.
    #     - Tg: 2D array of temperature grid values for contour plotting.
    #     - cores: Dictionary containing core water mass points to be marked on the plot.
    #             Example: {"SAAIW": (34.9, 6.5), "GCW": (36.4, 22.3), "NASUW": (36.8, 22)}
    #     - cmap_name: Name of the color map to use for distinguishing datasets (default: 'viridis').

    #     Returns:
    #     - None
    #     """

    #     # Initialize the plot
    #     fig, ax = plt.subplots(figsize=(10, 8))

    #     # Plot sigma_theta contours
    #     cs = ax.contour(Sg, Tg, sigma_theta, colors='grey', zorder=1)

    #     # Create a color map
    #     cmap = plt.get_cmap(cmap_name)
    #     colors = cmap(np.linspace(0, 1, len(datasets)))

    #     # Plot T-S profiles for each dataset
    #     for idx, (TEMP, SAL) in enumerate(datasets):
    #         label = dataset_labels[idx]
    #         color = colors[idx]

    #         # Ensure TEMP and SAL are 2D arrays for plotting
    #         if TEMP.ndim == 1:
    #             TEMP = TEMP[:, np.newaxis]
    #         if SAL.ndim == 1:
    #             SAL = SAL[:, np.newaxis]

    #         for i in range(TEMP.shape[1]):  # Plot each profile in the dataset
    #             ax.plot(SAL[:, i], TEMP[:, i], color=color, linewidth=0.5, label=label if i == 0 else "")

    #     # Plot core water masses
    #     for label, (salinity, temperature) in cores.items():
    #         ax.plot(salinity, temperature, 'o', markersize=7, color='black')
    #         ax.text(salinity, temperature, label, fontsize=11, verticalalignment='bottom', horizontalalignment='right', fontweight='bold')

    #     # Configure the plot
    #     ax.set_xlim(34.5, 37.5)
    #     plt.clabel(cs, fontsize=10, inline=False, fmt='%.1f', colors='k')
    #     plt.xlabel('Salinity [PSU]')
    #     plt.ylabel('Temperature [°C]')
    #     plt.title('T-S Diagram')
    #     plt.legend(fontsize=10)
    #     plt.show()

    # def index_for_range(data, min_val, max_val):
    #     return np.where((data >= min_val) & (data <= max_val))[0]

    # # Filter data based on SSH ranges
    # ssh_nature_run = ssh_nature_run.flatten()  # Assuming SSH values need to be compared
    # # Remove NaN values and corresponding indices from b, T_nature_run, and S_nature_run
    # valid_indices = ~np.isnan(ssh_nature_run)
    # ssh_nature_run = ssh_nature_run[valid_indices]
    # T_nature_run = T_nature_run[valid_indices]
    # S_nature_run = S_nature_run[valid_indices]
    # ssh_05to_01 = index_for_range(ssh_nature_run, -0.05, -0.01)
    # ssh_01to01 = index_for_range(ssh_nature_run, -0.01, 0.01)
    # ssh_01to10 = index_for_range(ssh_nature_run, 0.01, 0.1)
    # ssh_10to30 = index_for_range(ssh_nature_run, 0.1, 0.3)

    # # Build datasets with correct dimensions
    # datasets = [
    #     (T_nature_run[ssh_05to_01], S_nature_run[ssh_05to_01]),
    #     (T_nature_run[ssh_01to01], S_nature_run[ssh_01to01]),
    #     (T_nature_run[ssh_01to10], S_nature_run[ssh_01to10]),
    #     (T_nature_run[ssh_10to30], S_nature_run[ssh_10to30])
    # ]

    # dataset_labels = ['SSH -0.05 to -0.01', 'SSH -0.01 to 0.01', 'SSH 0.01 to 0.1', 'SSH 0.1 to 0.3']

    # # Plotting the T-S profiles
    # plot_ts_profiles(datasets, dataset_labels, sigma_theta, Sg, Tg, cores, cmap_name='viridis')
