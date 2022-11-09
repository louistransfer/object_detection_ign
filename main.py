import os
from object_detection_ign.satellite_view import WMTSClient

# WMTS_SERVICE_URL = "https://wxs.ign.fr/satellite/geoportail/wmts"
WMTS_SERVICE_URL = "https://wxs.ign.fr/ortho/geoportail/wmts?SERVICE=WMTS"
CORRESPONDANCE_TABLE_URL = "https://developers.arcgis.com/documentation/mapping-apis-and-services/reference/zoom-levels-and-scale/"
DATA_PATH = os.path.join("data")
CORRESPONDANCE_TABLE_PATH = os.path.join(DATA_PATH, "correspondance_table.csv")

client = WMTSClient(
    WMTS_SERVICE_URL, CORRESPONDANCE_TABLE_PATH, CORRESPONDANCE_TABLE_URL
)

print(client.list_available_zoom_options())
print(client.list_available_layers())
satellite_view = client.create_satellite_view_from_address(
    "Aérodrome de Carpiquet, France", "HR.ORTHOIMAGERY.ORTHOPHOTOS", 15
)
satellite_view.show_image()