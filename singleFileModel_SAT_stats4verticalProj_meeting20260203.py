# %%
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from nespreso.analysis import (
    average_depth,
    bin_data,
    calculate_correlation,
    compute_depth_interval_metrics,
    compute_depth_rmse_bias,
    compute_profile_residual,
    compute_season_masked_depth_rmse_bias,
    default_depth_intervals,
    equivalent_average_statistic,
    fit_pcs_regression_exact_gpu,
    get_glider_predictions,
    histogram_available_depths,
    isop_depth_indices,
    predict_pcs_exact_gpu,
    prepare_features,
)
from nespreso.analysis.density import (
    compute_density_profiles,
    compute_smoothness_metrics,
    compute_stability_metrics,
)
from nespreso.data.dataset import TemperatureSalinityDataset
from nespreso.data.features import prepare_inputs
from nespreso.data.pca import sklearn_inverse_transform_pcs
from nespreso.data.splits import IndexedSubset, split_dataset
from nespreso.determinism import get_device, set_seed
from nespreso.experiments.density_stability import run_density_stability
from nespreso.experiments.depth_interval_stats import run_depth_interval_stats
from nespreso.experiments.glider_mission import run_glider_mission
from nespreso.experiments.monthly_distribution import run_monthly_distribution
from nespreso.experiments.pca_regression import run_pca_regression_baseline
from nespreso.experiments.steric_depth_stats import run_steric_depth_stats
from nespreso.experiments.validation_context import build_validation_context
from nespreso.experiments.validation_maps import run_validation_maps
from nespreso.inference import (
    get_inputs,
    get_predictions,
    get_predictions_torchscript,
    load_all_models,
    predict_with_numpy,
)
from nespreso.io.argo import load_argo_mat
from nespreso.io.satellite import load_satellite_data, load_satellite_data_for_dataset
from nespreso.io.satellite_readers import get_aviso_by_date
from nespreso.losses import (
    CombinedPCALoss,
    PCALoss,
    WeightedMSELoss,
    genWeightedMSELoss,
    make_loss,
)
from nespreso.metrics import bias, mad, rmse
from nespreso.models.density import DensityConstraint, RhoMLP
from nespreso.models.mlp import PredictionModel
from nespreso.physics_metrics import (
    density_smoothness_metrics,
    eos_from_SP_T,
    second_derivative,
    static_stability_metrics,
)
from nespreso.reporting import print_training_params
from nespreso.train import evaluate_model, train_model
from nespreso.utils.geo import calculate_distances, haversine
from nespreso.utils.time import datenum_to_datetime, datenums_to_datetimes, get_month, get_season, matlab2datetime
from nespreso.viz.coefficients import plot_coefficients_heatmap
from nespreso.viz.fields import plot_field, plot_field_subplot
from nespreso.viz.maps import (
    calculate_average_in_bin,
    plot_bin_map,
    plot_comparison_maps,
    plot_residual_profiles_for_top_bins,
    plot_rmse_on_ax,
)
from nespreso.viz.profiles import (
    calculate_bias,
    filter_by_season,
    seasonal_plots,
    visualize_combined_results,
)

plt.rcParams.update({"font.size": 18})
load_trained_model = False
ensemble_models = False
load_dataset_file = True
gen_paula_profiles = False
global debug
debug = False  # Set to False to disable debugging
seed = 42
n_runs = 1  # number of model runs
nn_repeat_time = 10  # number of nespreso runs for generation timing
gem_repeat_time = 1  # number of GEM runs for generation timing

set_seed(seed)
DEVICE = get_device()

coolwhitewarm = mcolors.LinearSegmentedColormap.from_list(
    name="red_white_blue", colors=[(0, 0, 1), (1, 1.0, 1), (1, 0, 0)]
)


def inverse_transform(pcs, pca_temp, pca_sal, n_components):
    return sklearn_inverse_transform_pcs(pcs, pca_temp, pca_sal, n_components)


# %%
if __name__ == "__main__":
    from nespreso.config import load_config
    from nespreso.runner import run_training

    cfg = load_config()
    bin_size = 1  # bin size in degrees (monolith-only visualization knob)

    _save_model_path, artifacts = run_training(cfg, return_artifacts=True)
    ctx = build_validation_context(cfg, artifacts, bin_size=bin_size)

    run_steric_depth_stats(ctx)
    run_pca_regression_baseline(ctx)
    run_validation_maps(ctx)
    run_glider_mission(ctx)
    run_depth_interval_stats(ctx)
    run_density_stability(ctx)
    run_monthly_distribution(ctx)
