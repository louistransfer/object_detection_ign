import os
import toml
from starlite import State
from requests.exceptions import HTTPError, Timeout, ConnectionError
from starlite.exceptions import ServiceUnavailableException

from object_detection_ign.inference_helpers import load_inference_model
from object_detection_ign.satellite_view import WMTSClient

def set_state_on_startup(state: State) -> None:
    """Loads a toml config file, reads its parameters and assign them to a Starlite State object.

    Args:
        state (State): a Starlite State object
    """
    config = toml.load(state.config_file_path)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = config["model"]["TF_CPP_MIN_LOG_LEVEL"]

    state.DATA_PATH: str = config["wmts"]["DATA_PATH"]
    state.WMTS_SERVICE_URL: str = config["wmts"]["WMTS_SERVICE_URL"]
    state.CORRESPONDANCE_TABLE_URL: str = config["wmts"]["CORRESPONDANCE_TABLE_URL"]
    state.CORRESPONDANCE_TABLE_FILE: str = os.path.join(
        state.DATA_PATH, config["wmts"]["CORRESPONDANCE_TABLE_FILE"]
    )
    state.MODEL_PATH: str = config["model"]["MODEL_PATH"]
    state.CLASSES_DICT: dict = {
        int(key): value for key, value in config["model"]["classes_dict"].items()
    }


def api_initialization(state: State):
    """Initializes the API on startup by loading the inference model and the WMTS client.

    Args:
        state (State): a Starlite State object

    Raises:
        ServiceUnavailableException: errors are raised when the WMTS Server is unreachable
    """
    if not getattr(state, "inference_model", None):
        (
            state.inference_model,
            state.input_img_width,
            state.input_img_height,
        ) = load_inference_model(state.MODEL_PATH)
    if not getattr(state, "wmts_server", None):
        try:
            state.wmts_client = WMTSClient(
                state.WMTS_SERVICE_URL,
                state.CORRESPONDANCE_TABLE_FILE,
                state.CORRESPONDANCE_TABLE_URL,
            )
        except HTTPError:
            raise ServiceUnavailableException(
                detail=f"Error {HTTPError.errno} occured."
            )

        except Timeout:
            raise ServiceUnavailableException(
                detail="The connection with the WMTS Server timed out."
            )

        except ConnectionError:
            raise ServiceUnavailableException