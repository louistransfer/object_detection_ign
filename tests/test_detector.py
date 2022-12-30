from object_detection_ign.detector.inference_helpers import load_inference_model
import tensorflow as tf

def test_model_loading(model_definition):
    satellite_detector, input_img_width, input_img_height = load_inference_model(model_definition["model_path"])
    assert isinstance(satellite_detector, tf.lite.Interpreter)
    assert input_img_height == model_definition["input_img_width"]
    assert input_img_width == model_definition["input_img_height"]