from object_detection_ign.wmts.satellite_view import WMTSClient

def test_wmts_client(wmts_test_client: WMTSClient):
    assert len(wmts_test_client.available_options) > 0
    assert len(wmts_test_client.wmts_instance.contents) > 0
