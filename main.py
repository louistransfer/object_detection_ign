import os
import toml
import io

# from object_detection_ign.satellite_view import WMTSClient

# # WMTS_SERVICE_URL = "https://wxs.ign.fr/satellite/geoportail/wmts"
# WMTS_SERVICE_URL = "https://wxs.ign.fr/ortho/geoportail/wmts?SERVICE=WMTS"
# CORRESPONDANCE_TABLE_URL = "https://developers.arcgis.com/documentation/mapping-apis-and-services/reference/zoom-levels-and-scale/"
# DATA_PATH = os.path.join("data")
# CORRESPONDANCE_TABLE_PATH = os.path.join(DATA_PATH, "correspondance_table.csv")
from starlite import Starlite, get, post, State, MediaType
from starlite.exceptions import (
    HTTPException,
    ValidationException,
    ServiceUnavailableException,
)
from starlite.status_codes import HTTP_404_NOT_FOUND, HTTP_503_SERVICE_UNAVAILABLE

from starlite.controller import Controller
from object_detection_ign.satellite_view import WMTSClient, SatelliteView
from object_detection_ign.inference_helpers import (
    load_inference_model,
    perform_inference,
)
from object_detection_ign.api import SatelliteAdress, SatellitePosition
from logzero import logger
from requests.exceptions import HTTPError, Timeout, ConnectionError

CONFIG_FILE_PATH = os.path.join("config", "api_config.toml")
state = State()


def set_state_on_startup(state: State) -> None:
    """Startup and shutdown hooks can receive `State` as a keyword arg."""
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


class ObjectDetectionController(Controller):
    path = "/inference"

    @post("/address", media_type="image/png")
    def detect_objects_address(self, data: SatelliteAdress, state: State) -> bytearray:

        satellite_view: SatelliteView = (
            state.wmts_client.create_satellite_view_from_address(
                data.address, data.layer, data.zoom_level
            )
        )
        logger.info("Found coordinates ?: ", satellite_view.found_coordinates)
        if satellite_view.found_coordinates:
            satellite_view.crop_image_center(
                state.input_img_width, state.input_img_height
            )
            scores, labels, bounding_boxes = perform_inference(
                state.inference_model,
                satellite_view,
                state.CLASSES_DICT,
                detection_threshold=0.1,
            )

            logger.info("Inference performed.")

            img_byte_arr = io.BytesIO()
            satellite_view.image.save(img_byte_arr, format="PNG")
            img_byte_arr = img_byte_arr.getvalue()
            return img_byte_arr
        else:
            logger.critical(
                "The requested address was not found, try to change it slightly or use the coordinates endpoint."
            )
            raise ValidationException(
                detail="The requested address was not found in OpenStreetMap, try to change it slightly or use the coordinates endpoint."
            )

    @post("/location", media_type="image/png")
    def detect_objects_location(
        self, data: SatellitePosition, state: State
    ) -> bytearray:
        satellite_view: SatelliteView = (
            state.wmts_client.create_satellite_view_from_location(
                data.latitude, data.longitude, data.layer, data.zoom_level
            )
        )

        if satellite_view.found_coordinates:
            satellite_view.crop_image_center(
                state.input_img_width, state.input_img_height
            )
            scores, labels, bounding_boxes = perform_inference(
                state.inference_model,
                satellite_view,
                state.CLASSES_DICT,
                detection_threshold=0.1,
            )

            logger.info("Inference performed.")

            img_byte_arr = io.BytesIO()
            satellite_view.image.save(img_byte_arr, format="PNG")
            img_byte_arr = img_byte_arr.getvalue()
            return img_byte_arr
        else:
            logger.critical(
                "The requested location was not found, check that the latitude and longitude are correct, or that the position is located in France."
            )
            raise ValidationException(
                detail="The requested location was not found, check that the latitude and longitude are correct, or that the position is located in France."
            )


app = Starlite(
    route_handlers=[ObjectDetectionController],
    on_startup=[
        set_state_on_startup,
        api_initialization,
    ],
    initial_state={"config_file_path": CONFIG_FILE_PATH},
)
