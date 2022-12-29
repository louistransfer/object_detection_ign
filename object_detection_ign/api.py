import os
from pydantic import BaseModel, UUID4, BaseSettings


class SatelliteAdress(BaseModel):
    address: str
    zoom_level: int
    layer: str


class SatellitePosition(BaseModel):
    longitude: float
    latitude: float
    zoom_level: int
    layer: str
