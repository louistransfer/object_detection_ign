from pyproj import CRS, Transformer
from owslib.wmts import TileMatrix, TileMatrixSet

def _convert_coordinates(longitude: float, latitude: float):
    gps_crs = CRS("EPSG:4326")
    mercator_crs = CRS("EPSG:3857")
    coordinates_transformer = Transformer.from_crs(gps_crs, mercator_crs, always_xy=True)
    x, y = coordinates_transformer.transform(longitude, latitude)
    return x, y

def compute_tile_position(
    matrix_set: TileMatrixSet, zoom_level: int, longitude: float, latitude: float
):
    # TileMatrixSet.tilematrix is a dictionnary containing the matrices !
    matrix_set_dict: dict[str, TileMatrix] = matrix_set.tilematrix
    target_matrix = matrix_set_dict[str(zoom_level)]
    tile_width_meters = target_matrix.scaledenominator * 0.00028 * 256
    print(f"Tile width in meters : {tile_width_meters}")
    x0, y0 = map(lambda x: float(x), target_matrix.topleftcorner)
    x, y = _convert_coordinates(longitude=longitude, latitude=latitude)
    print(f"X = {x}, Y = {y}, identifier = {target_matrix.identifier}")
    tile_col, tile_row = (x - x0) / tile_width_meters, (y0 - y) / tile_width_meters
    tile_col, tile_row = int(round(tile_col, 0)), int(round(tile_row, 0))
    return tile_row, tile_col