"""ARCHIVED — monolith __main__ block (Phase 9 removal, commit 2cbb2b1).

Source: legacy/monolith/singleFileModel_SAT_stats4verticalProj_meeting20260203.py
lines ~329-375. Nature-run SSH histogram.

NOT wired. Code below is preserved verbatim (still commented).
"""

    # ## Eddy experiment - nature run stuff

    # #compare ssh distributions:

    # from matplotlib.ticker import PercentFormatter
    # from scipy.io import loadmat

    # def aggregate_from_mat(folder_path, *variable_names):
    #     aggregated_data = {var_name: [] for var_name in variable_names}

    #     # Loop through all files in the directory
    #     for filename in os.listdir(folder_path):
    #         if filename.endswith('.mat'):
    #             file_path = os.path.join(folder_path, filename)
    #             mat_data = loadmat(file_path)

    #             # Check if each variable exists in the .mat file and aggregate
    #             for var_name in variable_names:
    #                 if var_name in mat_data:
    #                     var_data = mat_data[var_name]
    #                     aggregated_data[var_name].append(np.expand_dims(var_data, axis=-1))
    #                 else:
    #                     print(f"'{var_name}' not found in {filename}")

    #     # Combine all variable data into single numpy arrays along the new axis
    #     for var_name in variable_names:
    #         if aggregated_data[var_name]:
    #             aggregated_data[var_name] = np.concatenate(aggregated_data[var_name], axis=-1)
    #         else:
    #             print(f"No '{var_name}' data found in any .mat files.")

    #     return aggregated_data

    # # Example usage:
    # folder_path = '/unity/g2/jmiranda/SubsurfaceFields/Data/NatureRun/'
    # ssh_nature_run = aggregate_from_mat(folder_path, 'ssh10')['ssh10'].flatten()

    # a = full_dataset.AVISO_ADT
    # n, bins, _ = plt.hist(a, weights=np.ones(len(a))/len(a), bins=100, color='blue', label='Training AVISO SSH')
    # plt.hist(ssh_nature_run, weights=np.ones(len(ssh_nature_run))/len(ssh_nature_run), bins=bins, color='red', label='Nature run SSH')
    # plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

    # # Set custom x-ticks every 0.1 from -0.4 to 0.9
    # plt.xticks(np.arange(-0.4, 1.0, 0.1), fontsize=11)
    # plt.yticks(fontsize=11)
    # plt.legend(fontsize=11)
