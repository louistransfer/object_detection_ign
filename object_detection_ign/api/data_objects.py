from pydantic import BaseModel


class SatelliteAddress(BaseModel):
    address: str
    zoom_level: int
    layer: str


class SatellitePosition(BaseModel):
    longitude: float
    latitude: float
    zoom_level: int
    layer: str
