from starlite.status_codes import HTTP_200_OK, HTTP_201_CREATED
from starlite.testing import TestClient


def test_health_check(test_client: TestClient):
    with test_client as client:
        response = client.get("/health")
        assert response.status_code == HTTP_200_OK
        assert response.text == "healthy"


def test_address(test_client: TestClient, address_data: dict):
    with test_client as client:
        response = client.post("/inference/address", json=address_data)
        assert response.status_code == HTTP_201_CREATED
        assert type(response.content) == bytes


def test_location(test_client: TestClient, location_data: dict):
    with test_client as client:
        response = client.post("/inference/location", json=location_data)
        assert response.status_code == HTTP_201_CREATED
        assert type(response.content) == bytes
