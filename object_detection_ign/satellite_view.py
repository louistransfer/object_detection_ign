import os
import io
import pandas as pd
import requests
from PIL import Image
from owslib.wmts import WebMapTileService, TileMatrixSet
from logzero import logger
from object_detection_ign.utils import compute_tile_position
from tqdm import trange


class SatelliteView:
    def __init__(self):
        self.latitude = 0.0
        self.longitude = 0.0
        self.zoom_level = 0
        self.address = None
        self.image: Image.Image = None

    def show_image(self):
        self.image.show()

    def save_image(self, export_path: str):
        self.image.save(export_path, "PNG")

    def crop_image_center(self, new_width: int, new_length: int):
        old_width, old_length = self.image.size
        x_center, y_center = old_width // 2, old_length // 2
        new_left = x_center - new_width // 2
        new_upper = y_center - new_length // 2
        new_right = x_center + new_width // 2
        new_lower = y_center + new_length // 2
        self.image = self.image.crop([new_left, new_upper, new_right, new_lower])
        logger.info(
            f"Resized image from size {old_width, old_length} to size {self.image.size}."
        )


class WMTSClient:
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

    def list_available_zoom_options(self):
        return self.available_options

    def list_available_layers(self):
        return self.wmts_instance.contents.keys()

    def reverse_geocoding(self, address: str):
        """This functions performs an API call with the target address to the OpenStreetMap API.
        A longitude and a latitude are then returned."""

        target_url = (
            "https://nominatim.openstreetmap.org/search?q=" + address + "&format=json"
        )

        target_url = target_url.replace(",", "%2C")
        target_url = target_url.replace(" ", "+")

        r = requests.get(target_url)
        found_coordinates = True
        if r.status_code == 200:
            try:
                coordinates = r.json()[0]
                latitude, longitude = coordinates["lat"], coordinates["lon"]

            except:
                print(f"Address not found for: {target_url}")
                latitude, longitude = None, None
                found_coordinates = False

        else:
            print(f"Error {r.status_code} ocurred on the request")
            latitude, longitude, found_coordinates = None, None, False
        return latitude, longitude, found_coordinates

    def get_concat_image(
        self, grid_length: int, grid_width: int, tile_row: int, tile_column: int, layer: str, zoom_level: int
    ):
        total_width = 256 * grid_width
        total_height = 256 * grid_length
        dst = Image.new("RGB", (total_width, total_height))
        for i, row in enumerate(trange(tile_row - 1, tile_row + 2)):
            for j, col in enumerate(range(tile_column - 1, tile_column + 2)):
                request = self.wmts_instance.gettile(
                    layer=layer,
                    tilematrixset="PM",
                    tilematrix=zoom_level,
                    row=row,
                    column=col,
                )
                temp_img = Image.open(io.BytesIO(request.read()))
                dst.paste(temp_img, (j * 256, i * 256))
        return dst

    def create_satellite_view_from_address(
        self, address, layer, zoom_level, grid_length=3, grid_width=3
    ):
        satellite_view = SatelliteView()
        satellite_view.address = address
        satellite_view.zoom_level = zoom_level
        (
            satellite_view.latitude,
            satellite_view.longitude,
            found_coordinates,
        ) = self.reverse_geocoding(address)
        tile_row, tile_column = compute_tile_position(
            self.matrix_set,
            zoom_level,
            satellite_view.longitude,
            satellite_view.latitude,
        )
        satellite_view.image = self.get_concat_image(
            grid_length, grid_width, tile_row, tile_column, layer, zoom_level
        )
        return satellite_view

    def create_satellite_view_from_position(
        self, latitude, longitude, layer, zoom_level, grid_length=3, grid_width=3
    ):
        satellite_view = SatelliteView()
        satellite_view.latitude = latitude
        satellite_view.longitude = longitude
        tile_row, tile_column = compute_tile_position(
            self.matrix_set, zoom_level, longitude, latitude
        )
        satellite_view.image = self.get_concat_image(
            grid_length, grid_width, tile_row, tile_column, layer, zoom_level
        )
        return satellite_view
