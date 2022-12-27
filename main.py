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
from starlite.exceptions import HTTPException
from starlite.status_codes import HTTP_404_NOT_FOUND, HTTP_503_SERVICE_UNAVAILABLE
from pydantic import BaseModel, UUID4, BaseSettings
from starlite.controller import Controller
from object_detection_ign.satellite_view import WMTSClient, SatelliteView
from object_detection_ign.inference_helpers import (
    load_inference_model,
    perform_inference,
)
from logzero import logger
from requests.exceptions import HTTPError, Timeout, ConnectionError

# from starlite.types import Partial

# class ObjectDetection(BaseModel):
#     user_id: int
#     order: str

CONFIG_FILE_PATH = os.path.join("config", "api_config.toml")
config = toml.load(CONFIG_FILE_PATH)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = config["model"]["TF_CPP_MIN_LOG_LEVEL"]


class AddressNotFound(HTTPException):
    """The requested address was not found on OpenStreetMap. Please try another syntax or use the coordinates endpoint."""

    def __init__(
        self,
        detail="The requested address was not found on OpenStreetMap. Please try another syntax or use the coordinates endpoint. ",
    ):
        super().__init__(detail)
        self.status_code = HTTP_404_NOT_FOUND
        self.detail = detail


class WMTSUnreachable(HTTPException):
    """The requested address was not found on OpenStreetMap. Please try another syntax or use the coordinates endpoint."""

    def __init__(
        self, detail="The WTMS Server is currently unreachable, please try out later."
    ):
        super().__init__(detail)
        self.status_code = HTTP_503_SERVICE_UNAVAILABLE
        self.detail = detail

class AppSettings(BaseSettings):
    DATA_PATH: str = config["wmts"]["DATA_PATH"]
    WMTS_SERVICE_URL: str = config["wmts"]["WMTS_SERVICE_URL"]

    CORRESPONDANCE_TABLE_URL: str = config["wmts"]["CORRESPONDANCE_TABLE_URL"]
    CORRESPONDANCE_TABLE_FILE: str = os.path.join(
        DATA_PATH, config["wmts"]["CORRESPONDANCE_TABLE_FILE"]
    )
    MODEL_PATH: str = config["model"]["MODEL_PATH"]
    CLASSES_DICT: dict = {
        int(key): value for key, value in config["model"]["classes_dict"].items()
    }


settings = AppSettings()

print(settings.CLASSES_DICT[1])


class Satellite(BaseModel):
    address: str
    zoom_level: int
    layer: str


def api_initialization(state: State):
    if not getattr(state, "inference_model", None):
        (
            state.inference_model,
            state.input_img_width,
            state.input_img_height,
        ) = load_inference_model(settings.MODEL_PATH)
    if not getattr(state, "wmts_server", None):
        try:
            state.wmts_client = WMTSClient(
                settings.WMTS_SERVICE_URL,
                settings.CORRESPONDANCE_TABLE_FILE,
                settings.CORRESPONDANCE_TABLE_URL,
            )
        except HTTPError:
            raise WMTSUnreachable(detail=f"Error {HTTPError.errno} occured.")

        except Timeout:
            raise WMTSUnreachable

        except ConnectionError:
            raise WMTSUnreachable


class ObjectDetectionController(Controller):
    path = "/inference"

    @post("/address", media_type="image/png")
    def detect_objects_address(self, data: Satellite, state: State) -> dict[str, str]:

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
                settings.CLASSES_DICT,
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
            raise AddressNotFound

    @post("/location")
    def detect_objects_location(self) -> dict[str, str]:
        """Handler function that returns a greeting dictionary."""
        return {"a": "b"}


app = Starlite(
    route_handlers=[ObjectDetectionController], on_startup=[api_initialization]
)


# client = WMTSClient(
#     WMTS_SERVICE_URL, CORRESPONDANCE_TABLE_PATH, CORRESPONDANCE_TABLE_URL
# )

# print(client.list_available_zoom_options())
# print(client.list_available_layers())
# # satellite_view = client.create_satellite_view_from_address(
# #     "Carrefour avenue de l'Europe, Venette, France", "HR.ORTHOIMAGERY.ORTHOPHOTOS", 16
# # )
# satellite_view = client.create_satellite_view_from_position(49.001190, 2.577033, "HR.ORTHOIMAGERY.ORTHOPHOTOS", 17)

# satellite_view.save_image("test_img.png")
