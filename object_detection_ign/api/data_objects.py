from pydantic import BaseModel


class SatelliteAddress(BaseModel):
    """Data model for SatelliteView objects generated through an address input.

    Args:
        BaseModel (_type_): a Starlite BaseModel
    """

    address: str
    zoom_level: int = 19
    layer: str = "HR.ORTHOIMAGERY.ORTHOPHOTOS"


class SatellitePosition(BaseModel):
    """Data model for SatelliteView objects generated through coordinates input.

    Args:
        BaseModel (_type_): a Starlite BaseModel
    """

    longitude: float
    latitude: float
    zoom_level: int = 19
    layer: str = "HR.ORTHOIMAGERY.ORTHOPHOTOS"
