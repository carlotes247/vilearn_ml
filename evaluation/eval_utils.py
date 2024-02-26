import mlflow
import numpy as np
from typing import Tuple
from mlflow.tracking import MlflowClient
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from scipy.stats import wasserstein_distance
from sklearn.neighbors import KernelDensity
from constant import RANDOM_SEED


def dimensionality_reduction(samples: np.ndarray, generated_samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform dimensionality reduction using PCA.

    Args:
        samples: A numpy array representing the original samples.
        generated_samples: A numpy array representing the generated samples.

    Returns:
        A tuple of numpy array containing the reduced original samples and the reduced generated samples.
    """
    # use PCA to reduce the dimensionality of the data.
    pca = PCA(n_components=15, random_state=RANDOM_SEED)  # 15D vector preserves more than 95 percent of the variance
    pca.fit(samples)
    reduced_generated = pca.transform(generated_samples)
    reduced_real = pca.transform(samples)
    return reduced_real, reduced_generated


def compute_log_likelihood(samples: np.ndarray, generated_samples: np.ndarray) -> list:
    """
    Compute the log-likelihood of generated samples using Parzen windows.

    Args:
        samples: A numpy array representing the original samples.
        generated_samples: A numpy array representing the generated samples.

    Returns:
        A list representing the log-likelihood values.
    """
    # use Parzen windows to estimate the probability distribution of the generated samples.
    # use cross-validation to set the bandwidth of the Parzen estimator.
    kde = GridSearchCV(KernelDensity(kernel='gaussian'), {'bandwidth': np.linspace(-1.0, 1.0, 30)}, cv=10)
    # fit KDE to the generated landmarks
    kde.fit(samples)

    # estimate the log-likelihood of the test samples from the estimated distribution.
    log_likelihood = kde.score_samples(generated_samples)
    return log_likelihood


def compute_rmse(samples: np.ndarray, generated_samples: np.ndarray) -> list:
    """
    Compute the Root Mean Squared Error (RMSE) between samples and generated samples.

    Args:
        samples: A numpy array representing the original samples.
        generated_samples: A numpy array representing the generated samples.

    Returns:
        A list representing the RMSE values.
    """
    rms = np.sqrt(np.mean((samples - generated_samples) ** 2, axis=-1))
    return rms


def compute_wasserstein_distance(samples: np.ndarray, generated_samples: np.ndarray) -> list:
    """
    Compute the Wasserstein distance between samples and generated samples.

    Args:
        samples: A numpy array representing the original samples.
        generated_samples: A numpy array representing the generated samples.

    Returns:
        A list of Wasserstein distance values.
    """
    distances = []
    for s, g in zip(samples, generated_samples):
        d = wasserstein_distance(s, g)
        distances.append(d)
    return distances


def get_experiment(experiment_name: str) -> Tuple[MlflowClient, str]:
    """
    Get the MlflowClient instance and experiment ID for a given experiment name.

    Args:
        experiment_name: A string representing the name of the experiment.

    Returns:
        A tuple containing the MlflowClient instance and the experiment ID.
    """
    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    exp_id = exp.experiment_id
    return client, exp_id


def experiment_description(experiment_name: str) -> None:
    """
    Prints the description of an experiment.

    Args:
        experiment_name: A string representing the name of the experiment.

    Returns:
        None
    """
    client, exp_id = get_experiment(experiment_name)
    runs = mlflow.search_runs(exp_id)
    print(f'Description of experiment {experiment_name}:\n{runs.describe().to_string()}')


def get_metric_list(experiment_name: str, metric_key: str) -> list:
    """
    Retrieves a list of metric values for a given metric key from all runs in an experiment.

    Args:
         experiment_name: A string representing the name of the experiment.
         metric_key: A string representing the metric key to retrieve the values for.

    Returns:
        A list of metric values.
    """
    client, exp_id = get_experiment(experiment_name)
    runs_ids = mlflow.search_runs(exp_id)['run_id']
    metric_list = [client.get_metric_history(id, metric_key) for id in runs_ids]
    metrics = [metric[0].value for metric in metric_list]
    return metrics
