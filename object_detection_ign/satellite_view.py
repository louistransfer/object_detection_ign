import os
import pandas as pd
from owslib.wmts import WebMapTileService, TileMatrixSet
from logzero import logger


WMTS_SERVICE_URL = "https://wxs.ign.fr/satellite/geoportail/wmts"
CORRESPONDANCE_TABLE_URL = "https://developers.arcgis.com/documentation/mapping-apis-and-services/reference/zoom-levels-and-scale/"
DATA_PATH = os.path.join("data")
CORRESPONDANCE_TABLE_PATH = os.path.join(DATA_PATH, "correspondance_table.csv")


class SatelliteView:
    def __init__(self, location):
        self.latitude = 0
        self.longitude = 0
        self.location = location
        self.adress = None


class WMTSConnexion:
    def __init__(
        self, url: str, correspondance_table_path: str, correspondance_table_url: str
    ):
        self.wmts_server_url: str = url
        self.correspondance_table_url: str = correspondance_table_url
        self.wmts_instance = WebMapTileService(self.wmts_server_url, version="1.0.0")
        self.matrix_set: TileMatrixSet = self.wmts_instance.tilematrixsets["PM"]
        self.available_options = self._load_available_options(correspondance_table_path)

    def _load_available_options(self, correspondance_table_path: str):
        if os.path.exists(correspondance_table_path):
            logger.info("Existing correspondance table found, loading it now.")
            correspondance_table = pd.read_csv(correspondance_table_path, sep=";")
        else:
            logger.warning(
                "No correspondance table found, building one now from the given url."
            )
            correspondance_table = pd.read_html(self.correspondance_table_url)[0]
            correspondance_table["Usage suggestion"] = correspondance_table[
                "Usage suggestion"
            ].fillna(method="ffill")
            correspondance_table["New legend"] = (
                correspondance_table["Zoom level"].astype(str)
                + " - "
                + correspondance_table["Scale"].astype(str)
                + " - "
                + correspondance_table["Usage suggestion"]
            )
            correspondance_table = correspondance_table.set_index("Zoom level")
            correspondance_table.to_csv(correspondance_table_path, sep=";")

        available_zoom_levels = list(
            set(correspondance_table.index).intersection(
                set([int(key) for key in self.matrix_set.tilematrix.keys()])
            )
        )
        correspondance_table = correspondance_table.iloc[
            available_zoom_levels
        ].sort_index()
        available_options = correspondance_table.loc[:, "New legend"].to_list()
        return available_options


connexion = WMTSConnexion(
    WMTS_SERVICE_URL, CORRESPONDANCE_TABLE_PATH, CORRESPONDANCE_TABLE_URL
)
