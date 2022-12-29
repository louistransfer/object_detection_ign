import os
from starlite import Starlite
from object_detection_ign.api.routes import ObjectDetectionController

from object_detection_ign.api.api_configuration import (
    set_state_on_startup,
    api_initialization,
)
from logzero import logger


CONFIG_FILE_PATH = os.path.join("config", "api_config.toml")


app = Starlite(
    route_handlers=[ObjectDetectionController],
    on_startup=[
        set_state_on_startup,
        api_initialization,
    ],
    initial_state={"config_file_path": CONFIG_FILE_PATH},
)
