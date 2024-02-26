import torch
import mlflow
import random
import argparse
import numpy as np
import torch.nn as nn
from typing import Tuple, Union
from torch.utils.data import DataLoader
from models.MyDataset import MyDataset
from models.csg_utils import Generator, create_noise_vector
from models.lstm_utils import LSTM
from evaluation.fap_generation import create_generated_lip_movements_faps, create_video, smooth_landmarks
from evaluation.eval_utils import compute_rmse, dimensionality_reduction, compute_log_likelihood, \
    experiment_description, compute_wasserstein_distance
from feature_extraction.lm_preprocessing import remove_delay
from models.model_config import csg_config, lstm_config, model_dir, output_dim
from constant import VIDEO_PATH, ROOT_DIR, RANDOM_SEED, N_GPU, FRAME_DELAY, MOUTH_IDX, ANIMATION_PATH

# set random seed
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


def initialize_experiment(model: str, spectral_features: bool, emotional_speech_features: bool, remove_identity: bool,
                          add_context: bool, new_voice: bool) -> Tuple[str, str, str]:
    """
    Initialize an experiment and return the experiment name and ID.

    Args:
        model: A string representing the name of the model.
        spectral_features: A boolean indicating if spectral features are used.
        emotional_speech_features: A boolean indicating if emotional speech features are used.
        remove_identity: A boolean indicating if identity removal is applied.
        add_context: A boolean indicating if context is added.
        new_voice: A boolean indicating if the new voice data set is used for evaluation.

    Returns:
        A tuple of strings containing the experiment name, the mlrun experiment name and the experiment ID.
    """
    experiment_name = get_experiment_name(model, spectral_features, emotional_speech_features, remove_identity,
                                          add_context)
    if new_voice:
        mlrun_experiment_name = f'{experiment_name}_new_voice'
    else:
        mlrun_experiment_name = experiment_name
    experiment_id = create_experiment_id(mlrun_experiment_name)
    mlflow.set_experiment(experiment_id=experiment_id)
    return experiment_name, mlrun_experiment_name, experiment_id


def create_experiment_id(mlrun_experiment_name: str) -> str:
    """
    Create an experiment ID based on the experiment name.

    Args:
         mlrun_experiment_name: A string representing the name of the experiment.

    Returns:
         A string representing the experiment id.
    """
    mlflow_client = mlflow.tracking.MlflowClient()
    try:
        # creates a new experiment
        experiment_id = mlflow_client.create_experiment(mlrun_experiment_name)
    except(Exception,):
        experiment_description(mlrun_experiment_name)
        raise SystemExit
    print('Experiment Id: ', experiment_id)
    return experiment_id


def get_experiment_name(model: str, spectral_features: bool, emotional_speech_features: bool, remove_identity: bool,
                        add_context: bool) -> str:
    """
    Generate an experiment name based on the model and feature settings.

    Args:
        model: A string representing the name of the model.
        spectral_features: A boolean indicating if spectral features are used.
        emotional_speech_features: A boolean indicating if emotional speech features are used.
        remove_identity (bool): A boolean indicating if identity removal is applied.
        add_context (bool): A boolean indicating if context is added.

    Returns:
        A string representing the experiment name.
    """
    if spectral_features and not emotional_speech_features and not remove_identity and not add_context:
        experiment_name = f'{model}_sf'
    elif spectral_features and emotional_speech_features and not remove_identity and not add_context:
        experiment_name = f'{model}_sf_ef'
    elif spectral_features and emotional_speech_features and remove_identity and not add_context:
        experiment_name = f'{model}_sf_ef_ri'
    elif spectral_features and emotional_speech_features and not remove_identity and add_context:
        experiment_name = f'{model}_sf_ef_c'
    else:
        raise Exception('Model config not defined!')

    return experiment_name


def load_data(spectral_features: bool, emotional_speech_features: bool, add_context: bool, feature_path: str,
              data_set: str) -> DataLoader:
    """
    Load the data based on the feature settings.

    Args:
        spectral_features: A boolean indicating if spectral features are used.
        emotional_speech_features: A boolean indicating if emotional speech features are used.
        add_context: A boolean indicating if context is added.
        feature_path: A string specifying the directory path of the test data set
        data_set: A string specifying the test set which is loaded.

    Returns:
        The data loader object.
    """
    # do not remove identity information from test set
    data_set = MyDataset(video_path=VIDEO_PATH, feature_path=feature_path, root_dir=ROOT_DIR, mouth_idx=MOUTH_IDX,
                         spectral_features=spectral_features, emotional_speech_features=emotional_speech_features,
                         remove_identity=False, add_context=add_context, data_set=data_set)
    data_loader = DataLoader(data_set, batch_size=1, shuffle=False, num_workers=2)
    return data_loader


def load_model(model: str, experiment_name: str, device: torch.device) -> any:
    """
    Load the specified model based on the experiment name.

    Args:
        model: A string representing the name of the model.
        experiment_name: A string representing the name of the experiment.
        device: The device to load the model on.

    Returns:
        The loaded model (CSG Generator or LSTM).
    """
    if model == 'csg':
        generator = load_csg_model(experiment_name, device)
        return generator
    elif model == 'lstm':
        lstm = load_lstm_model(experiment_name, device)
        return lstm
    else:
        raise Exception(f'model must be "csg" or "lstm", got {model}')


def load_csg_model(experiment_name: str, device: torch.device) -> Generator:
    """
    Load the CSG Generator model based on the experiment name.

    Args:
        experiment_name: A string representing the name of the experiment.
        device: The device to load the model on.

    Returns:
        The loaded CSG Generator model.
    """
    # Load csg specific details
    model_config = csg_config[experiment_name]
    input_dim_g = model_config['input_dim_g']
    hidden_dim_g = model_config['hidden_dim_g']
    generator_name = model_config['generator_name']

    # initialize generator
    generator = Generator(input_dim_g, hidden_dim_g, output_dim).to(device)
    # handle multi-GPU if desired
    if (device.type == 'cuda') and (N_GPU > 1):
        generator = nn.DataParallel(generator, list(range(N_GPU)))
        generator.load_state_dict(torch.load(f'{model_dir}/{generator_name}'))
    else:
        generator.load_state_dict(torch.load(f'{model_dir}/{generator_name}', map_location=torch.device('cpu')))
    return generator


def load_lstm_model(experiment_name: str, device: torch.device) -> LSTM:
    """
    Load the LSTM model based on the experiment name.

    Args:
        experiment_name: A string representing the name of the experiment.
        device: The device to load the model on.

    Returns:
        The loaded LSTM model.
    """
    # Load lstm specific details
    model_config = lstm_config[experiment_name]
    input_dim = model_config['input_dim']
    hidden_dim = model_config['hidden_dim']
    dropout_rate = model_config['dropout_rate']
    model_name = model_config['model_name']

    # initialize lstm
    lstm = LSTM(input_dim, hidden_dim, output_dim, dropout_rate).to(device)
    # handle multi-GPU if desired
    if (device.type == 'cuda') and (N_GPU > 1):
        lstm = nn.DataParallel(lstm, list(range(N_GPU)))
        lstm.load_state_dict(torch.load(f'{model_dir}/{model_name}'))
    else:
        lstm.load_state_dict(torch.load(f'{model_dir}/{model_name}', map_location=torch.device('cpu')))
    return lstm


def generate_lip_movements(model: str, torch_model: any, speech_features: Union[torch.FloatTensor, torch.Tensor]) -> \
        Union[torch.FloatTensor, torch.Tensor]:
    """
    Generate lip movements using the specified model.

    Args:
          model: A string representing the name of the model.
          torch_model: The loaded torch model (CSG Generator or LSTM).
          speech_features: A tensor representing the input speech features.

    Returns:
        A tensor representing the generated lip movements.
    """
    if model == 'csg':
        z = create_noise_vector(speech_features)
        generated_lip_movements = torch_model(speech_features, z)
    elif model == 'lstm':
        generated_lip_movements = torch_model(speech_features)
    else:
        raise Exception(f'model must be "csg" or "lstm", got {model}')
    return generated_lip_movements


def compute_metrics(lip_movements: np.ndarray, generated_lip_movements: np.ndarray) -> Tuple[
    float, float, float, float, float, float]:
    """
    Compute various metrics between the original lip movements and generated lip movements.

    Args:
        lip_movements: A numpy array containing the original lip movements.
        generated_lip_movements: A numpy array containing the generated lip movements.

    Returns:
        A tuple containing the computed metrics.
    """
    # Compute all the necessary metrics
    rmse = compute_rmse(lip_movements, generated_lip_movements)
    average_rmse = np.mean(rmse).item()
    std_dev_rmse = np.std(rmse).item()

    reduced_real_lm, reduced_generated_lm = dimensionality_reduction(lip_movements, generated_lip_movements)

    log_likelihood = compute_log_likelihood(reduced_real_lm, reduced_generated_lm)
    average_log_likelihood = np.mean(log_likelihood).item()
    std_dev_log_likelihood = np.std(log_likelihood).item()

    wasserstein_distance = compute_wasserstein_distance(reduced_real_lm, reduced_generated_lm)
    average_wasserstein_distance = np.mean(wasserstein_distance).item()
    std_dev_wasserstein_distance = np.std(wasserstein_distance).item()

    return average_rmse, std_dev_rmse, average_log_likelihood, std_dev_log_likelihood, average_wasserstein_distance, std_dev_wasserstein_distance


def log_metrics(experiment_id: str, i: int, average_rmse: float, std_dev_rmse: float, average_log_likelihood: float,
                std_dev_log_likelihood: float, average_wasserstein_distance: float,
                std_dev_wasserstein_distance: float) -> None:
    """
    Log the computed metrics to Mlflow.

    Args:
        experiment_id: A string representing the ID of the experiment.
        i: An integer representing the index of the current run.
        average_rmse: A float representing the average RMSE value.
        std_dev_rmse: A float representing the standard deviation of RMSE.
        average_log_likelihood: A float representing the average log-likelihood value.
        std_dev_log_likelihood: A float representing the standard deviation of log-likelihood.
        average_wasserstein_distance: A float representing the average Wasserstein distance.
        std_dev_wasserstein_distance: A float representing the standard deviation of Wasserstein distance.

    Returns:
        None
    """
    mlflow.start_run(experiment_id=experiment_id, run_name=str(i))
    mlflow.log_metric('rmse', float(average_rmse))
    mlflow.log_metric('std_rmse', float(std_dev_rmse))
    mlflow.log_metric('log_likelihood', float(average_log_likelihood))
    mlflow.log_metric('std_log_likelihood', float(std_dev_log_likelihood))
    mlflow.log_metric('wasserstein_distance', float(average_wasserstein_distance))
    mlflow.log_metric('std_wasserstein_distance', float(std_dev_wasserstein_distance))
    mlflow.end_run()


def print_metrics(average_rmse: float, std_dev_rmse: float, average_log_likelihood: float,
                  std_dev_log_likelihood: float, average_wasserstein_distance: float,
                  std_dev_wasserstein_distance: float) -> None:
    """
    Print the computed metrics.

    Args:
        average_rmse: A float representing the average RMSE value.
        std_dev_rmse: A float representing the standard deviation of RMSE.
        average_log_likelihood: A float representing the average log-likelihood value.
        std_dev_log_likelihood: A float representing the standard deviation of log-likelihood.
        average_wasserstein_distance: A float representing the average Wasserstein distance.
        std_dev_wasserstein_distance: A float representing the standard deviation of Wasserstein distance.

    Returns:
        None
    """
    print("\nAverage Log-Likelihood: ", round(average_log_likelihood, 2))
    print("Standard Deviation of Log-Likelihood: ", round(std_dev_log_likelihood, 2))
    print("Average RMSE: ", round(average_rmse, 2))
    print("Standard Deviation of RMSE: ", round(std_dev_rmse, 2))
    print("Average Wasserstein-Distance: ", round(average_wasserstein_distance, 2))
    print("Standard Deviation of Wasserstein-Distance: ", round(std_dev_wasserstein_distance, 2))


def main(args):
    model = args.model
    spectral_features = args.spectral_features
    emotional_speech_features = args.emotional_speech_features
    remove_identity = args.remove_identity
    add_context = args.context

    generate_faps = args.faps
    new_voice = args.new_voice
    debug = args.debug

    experiment_name, mlrun_experiment_name, experiment_id = initialize_experiment(model, spectral_features,
                                                                                  emotional_speech_features,
                                                                                  remove_identity, add_context,
                                                                                  new_voice)

    device = torch.device('cuda:0' if (torch.cuda.is_available() and N_GPU > 0) else 'cpu')

    if new_voice:
        feature_path = 'feature_extraction/new_voice'
        data_set = 'new_voice'
        generate_faps = False
    else:
        feature_path = 'feature_extraction/extracted_features'
        data_set = 'test'

    data_loader = load_data(spectral_features, emotional_speech_features, add_context, feature_path, data_set)

    torch_model = load_model(model, experiment_name, device)
    print(torch_model)

    for i, (lip_movements, speech_features) in enumerate(data_loader):
        lip_movements = lip_movements.to(device)
        speech_features = speech_features.to(device)

        generated_lip_movements = generate_lip_movements(model, torch_model, speech_features)

        lip_movements = lip_movements.cpu().data.numpy().squeeze(0)
        generated_lip_movements = generated_lip_movements.cpu().data.numpy().squeeze(0)

        if add_context:
            generated_lip_movements = remove_delay(generated_lip_movements, FRAME_DELAY)

        average_rmse, std_dev_rmse, average_log_likelihood, std_dev_log_likelihood, average_wasserstein_distance, \
        std_dev_wasserstein_distance = compute_metrics(lip_movements, generated_lip_movements)

        log_metrics(experiment_id, i, average_rmse, std_dev_rmse, average_log_likelihood, std_dev_log_likelihood,
                    average_wasserstein_distance, std_dev_wasserstein_distance)

        if generate_faps:
            create_generated_lip_movements_faps(generated_lip_movements, i, ANIMATION_PATH, experiment_name,
                                                feature_path, debug)

        if debug:
            print_metrics(average_rmse, std_dev_rmse, average_log_likelihood, std_dev_log_likelihood,
                          average_wasserstein_distance, std_dev_wasserstein_distance)

    experiment_description(mlrun_experiment_name)
    print(f'ID: {experiment_id}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Add arguments to the parser
    parser.add_argument('-m', '--model', type=str, default='csg', help='Define model type. "csg" or "lstm"')
    parser.add_argument('-sf', '--spectral_features', action='store_true', default=True,
                        help='Enable spectral audio features')
    parser.add_argument('-ef', '--emotional_speech_features', action='store_true', default=False,
                        help='Enable emotional speech audio features')
    parser.add_argument('-ri', '--remove_identity', action='store_true', default=False,
                        help='Remove identity information from landmarks')
    parser.add_argument('-c', '--context', action='store_true', default=False, help='Add context to trainings data')

    parser.add_argument('-f', '--faps', action='store_true', default=False, help='Enable FAP files generation')
    parser.add_argument('-nv', '--new_voice', action='store_true', default=False,
                        help='Enable the new voice test data set')
    parser.add_argument('-dbg', '--debug', action='store_true', default=False, help='Enable debug mode')
    args = parser.parse_args()

    main(args)
