import os
from starlite import Starlite
from object_detection_ign.api.routes import ObjectDetectionController, health_check

from object_detection_ign.api.api_configuration import (
    set_state_on_startup,
    api_initialization,
)
import picologging as logging

logging.basicConfig()
logger = logging.getLogger()

CONFIG_FILE_PATH = os.path.join("config", "api_config.toml")


app = Starlite(
    route_handlers=[ObjectDetectionController, health_check],
    on_startup=[
        set_state_on_startup,
        api_initialization,
    ],
    initial_state={"config_file_path": CONFIG_FILE_PATH},
)
