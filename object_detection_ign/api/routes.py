import io
from starlite import post, State
from logzero import logger
from starlite.controller import Controller
from starlite.exceptions import ValidationException
from object_detection_ign.satellite_view import SatelliteView
from object_detection_ign.api.data_objects import SatelliteAddress, SatellitePosition
from object_detection_ign.inference_helpers import perform_inference


class ObjectDetectionController(Controller):
    """Inherits from the Controller class. This object is used to define the routes belonging to the "inference" branch.
    It abstracts the "address" and "location" endpoints of the API.

    Args:
        Controller (_type_): a Starlite Controller object
    """

    path = "/inference"

    @post("/address", media_type="image/png")
    def detect_objects_address(self, data: SatelliteAddress, state: State) -> bytearray:
        """Performs object detection on a location specified by an address. A SatelliteView object is created through a call to
        the OpenStreetMaps reverse geocoding API in order to obtain its coordinates. If the address is incorrect, an error is raised.

        Args:
            data (SatelliteAddress): a SatelliteAddress pydantic data object
            state (State): a Starlite State object, used to load various parameters (e.g. model filepath location)

        Raises:
            ValidationException: an error is raised when the address is not found on the OpenStreetMaps reverse geocoding API

        Returns:
            bytearray: raw array containing the inference image with bounding boxes encoded as a png
        """

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
        """Performs object detection on GPS coordinates (latitude and longitude). 

        Args:
            data (SatellitePosition): a SatellitePosition pydantic data object
            state (State): a Starlite State object, used to load various parameters (e.g. model filepath location)

        Raises:
            ValidationException: an error is raised when the speciifed latitude and longitude are incorrect (invalid values or a location which
            is not in France)

        Returns:
            bytearray: raw array containing the inference image with bounding boxes encoded as a png
        """
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
