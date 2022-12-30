import os
from pytest import fixture
from starlite.testing import TestClient
from object_detection_ign.wmts.satellite_view import WMTSClient
from main import app


# API fixtures
@fixture(scope="function")
def test_client() -> TestClient:
    return TestClient(app=app)


@fixture
def address_data() -> dict:
    return {
        "address": "E.Leclerc, 60290 Cauffry",
        "layer": "HR.ORTHOIMAGERY.ORTHOPHOTOS",
        "zoom_level": 19,
    }


@fixture
def location_data() -> dict:
    return {
        "latitude": 48.83980726885963,
        "longitude": -1.5490468522920273,
        "layer": "HR.ORTHOIMAGERY.ORTHOPHOTOS",
        "zoom_level": 19,
    }


@fixture
def wmts_client_config():
    return {
        "url": "https://wxs.ign.fr/ortho/geoportail/wmts?SERVICE=WMTS",
        "correspondance_table_path": os.path.join("data", "correspondance_table.csv"),
        "correspondance_table_url": "https://developers.arcgis.com/documentation/mapping-apis-and-services/reference/zoom-levels-and-scale/",
    }


@fixture
def wmts_test_client(wmts_client_config) -> WMTSClient:
    return WMTSClient(
        url=wmts_client_config["url"],
        correspondance_table_path=wmts_client_config["correspondance_table_path"],
        correspondance_table_url=wmts_client_config["correspondance_table_url"],
    )

@fixture
def model_definition():
    return {"model_path": os.path.join("models", "tflite_detector_v2", "model.tflite"), "input_img_width": 640, "input_img_height": 640}