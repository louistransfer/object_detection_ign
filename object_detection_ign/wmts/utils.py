from pyproj import CRS, Transformer
from owslib.wmts import TileMatrix, TileMatrixSet


def _convert_coordinates(longitude: float, latitude: float) -> tuple[float, float]:
    """Takes GPS coordinates (EPSG 4326) as input and converts them to the Mercator projection (EPSG 3857).
    This function is used to easily find coordinates on a GPS-enabled system such as Google Maps or OpenStreetMap, and
    to convert them to a format compatible with the WMTS format.

    Args:
        longitude (float): GPS longitude
        latitude (float): GPS latitude

    Returns:
        x (float): Mercator longitude
        y (float): Mercator latitude
    """
    gps_crs = CRS("EPSG:4326")
    mercator_crs = CRS("EPSG:3857")
    coordinates_transformer = Transformer.from_crs(
        gps_crs, mercator_crs, always_xy=True
    )
    x, y = coordinates_transformer.transform(longitude, latitude)
    return x, y


def compute_tile_position(
    matrix_set: TileMatrixSet, zoom_level: int, longitude: float, latitude: float
) -> tuple[int, int]:
    """Locates the tile containing the targeted location. It is located on a matrix set, i.e a grid containing multiples tiles
    defined by their row and column. The row and the column are obtained by converting the width and length in meters: this operation
    is performed by multiplicating by 0.00028 (number of meters represented by one pixel) and 256 (number of pixels along the width of the tile).
    Since the width and the length are equal in terms of pixels (256), only the width is computed. However in the future both may be 
    computed to accomodate WMTS servers using rectangular tiles.

    Args:
        matrix_set (TileMatrixSet): the WMTS matrix set. There is one matrix set per zoom level. It contains rows and columns
        zoom_level (int): zoom level to use on the WMTS server
        longitude (float): the longitude of the point contained in the desired tile
        latitude (float): the latitude of the point contained in the desired tile

    Returns:
        tile_row (int): the row of the desired tile in the matrix set
        tile_column (int): the column of the desired tile in the matrix set
    """

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
