import tensorflow as tf
import numpy as np
from PIL import ImageDraw, ImageFont, Image
from pycoral.utils.edgetpu import list_edge_tpus

# from tensorflow.lite.experimental import load_delegate
from logzero import logger
from platform import system
from matplotlib import font_manager
from object_detection_ign.satellite_view import SatelliteView


def decode_img(img_path, img_height, img_width, convert_to_dtype=None):
    img = tf.io.read_file(img_path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [img_height, img_width])
    if convert_to_dtype is not None:
        img = tf.image.convert_image_dtype(img, dtype=convert_to_dtype, saturate=False)
    img = tf.reshape(img, [1, img_height, img_width, 3])
    # Resize the image to the desired size
    return img


def filter_predictions(output, classes_dict, detection_threshold=0.1):
    labels, bounding_boxes, scores = (
        output["output_2"].astype(int).squeeze(),
        output["output_3"].squeeze(),
        output["output_1"].squeeze(),
    )
    chosen_idx = [
        idx
        for idx, score, label in zip(range(len(scores)), scores, labels)
        if score >= detection_threshold and label != 0
    ]
    scores = scores[chosen_idx]
    labels = labels[chosen_idx]
    labels = [classes_dict[label] for label in labels]
    bounding_boxes = bounding_boxes[chosen_idx]
    return scores, labels, bounding_boxes


def _draw_bbox(
    image: Image.Image,
    bbox: np.array,
    label: str,
    score,
    color="orange",
    use_normalized_coordinates=True,
):
    system_fonts = [
        font
        for font in font_manager.findSystemFonts(fontpaths=None, fontext="ttf")
        if "DejaVuSansMono.ttf" in font
    ]
    font = ImageFont.truetype(system_fonts[0], 15)
    ymin, xmin, ymax, xmax = (bbox[i] for i in range(0, 4))
    im_width, im_height = image.size
    if use_normalized_coordinates:
        (left, right, top, bottom) = (
            xmin * im_width,
            xmax * im_width,
            ymin * im_height,
            ymax * im_height,
        )
    else:
        (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
    text_width, text_height = font.getsize(label)
    # Each display_str has a top and bottom margin of 0.05x.
    total_display_str_height = (1 + 2 * 0.05) * text_height
    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = bottom + total_display_str_height
    margin = np.ceil(0.01 * text_height)

    draw = ImageDraw.Draw(image)
    draw.rectangle((left, top, right, bottom), outline="orange", width=2)
    draw.rectangle(
        [
            (left, text_bottom - text_height - 2 * margin),
            (left + text_width + 2 * margin, text_bottom),
        ],
        fill=color,
    )
    draw.text(
        (left + margin, text_bottom - text_height - margin),
        label + "/" + str(score),
        fill="black",
        font=font,
    )


def draw_bounding_boxes_on_image(
    image,
    boxes,
    scores=np.array([]),
    labels=np.array([]),
    color="red",
    thickness=4,
):
    """Draws bounding boxes on image.
    Args:
      image: a PIL.Image object.
      boxes: a 2 dimensional numpy array of [N, 4]: (ymin, xmin, ymax, xmax). The
        coordinates are in normalized format between [0, 1].
      color: color to draw bounding box. Default is red.
      thickness: line thickness. Default value is 4.
      display_str_list_list: list of list of strings. a list of strings for each
        bounding box. The reason to pass a list of strings for a bounding box is
        that it might contain multiple labels.
    Raises:
      ValueError: if boxes is not a [N, 4] array
    """
    boxes_shape = boxes.shape
    if len(boxes_shape) != 2 or boxes_shape[1] != 4:
        raise ValueError("Input must be of size [N, 4]")
    for i in range(boxes_shape[0]):
        _draw_bbox(
            image,
            boxes[i],
            labels[i],
            scores[i],
            color
            # thickness,
        )

def load_coral_tpus():
    platform_dict = {
        "Windows": "edgetpu.dll",
        "Linux": "libedgetpu.so.1",
        "Darwin": "libedgetpu.1.dylib",
    }
    available_tpus = list_edge_tpus()
    current_system = system()

    if current_system in platform_dict:
        logger.info("System compatible with the Coral TPU.")
        if len(available_tpus) > 0:
            logger.info(f"Found {len(available_tpus)} existing Coral TPU.")
            available_delegates = []
            try:
                for tpu in available_tpus:
                    available_delegates.append(
                        tf.lite.experimental.load_delegate(
                            platform_dict[current_system]
                        )
                    )
            except ValueError as e:
                logger.critical(f"Error {e} occured. Switching to CPU predictions.")
        else:
            logger.info("No Coral TPU has been found. Switching to CPU predictions.")
            available_delegates = []
    else:
        logger.info(
            "System is incompatible with the Edge TPU. Switching to CPU predictions."
        )
        available_delegates = []
    return available_delegates

def load_inference_model(model_path: str):
    satellite_detector = tf.lite.Interpreter(
        model_path, experimental_delegates=load_coral_tpus()
    )
    satellite_detector.allocate_tensors()

    input_img_shape = satellite_detector.get_input_details()[0]["shape_signature"][1:3]
    input_img_width, input_img_height = input_img_shape[0], input_img_shape[1]
    logger.info(
        f"The selected model uses a {(input_img_width, input_img_height)} size for input images."
    )
    return satellite_detector, input_img_width, input_img_height


def perform_inference(
    satellite_detector: tf.lite.Interpreter,
    satellite_view: SatelliteView,
    classes_dict: dict,
    detection_threshold=0.1,
):
    tf_img = tf.constant(satellite_view.image_array, dtype=tf.float32)
    signature = satellite_detector.get_signature_runner()
    output = signature(images=tf_img)
    scores, labels, bounding_boxes = filter_predictions(
        output, classes_dict, detection_threshold=detection_threshold
    )
    draw_bounding_boxes_on_image(satellite_view.image, bounding_boxes, scores, labels)
    return scores, labels, bounding_boxes


# def _draw_bounding_box_on_image(
#     image,
#     bounding_box,
#     color="red",
#     thickness=4,
#     display_str_list=(),
#     use_normalized_coordinates=True,
# ):
#     """Adds a bounding box to an image.
#     Bounding box coordinates can be specified in either absolute (pixel) or
#     normalized coordinates by setting the use_normalized_coordinates argument.
#     Each string in display_str_list is displayed on a separate line above the
#     bounding box in black text on a rectangle filled with the input 'color'.
#     If the top of the bounding box extends to the edge of the image, the strings
#     are displayed below the bounding box.
#     Args:
#       image: a PIL.Image object.
#       bounding_box: a numpy array with the format (top, left, bottom, right) i.e (ymin, xmin, ymax, xmax).
#       xmin: xmin of bounding box.
#       ymax: ymax of bounding box.
#       xmax: xmax of bounding box.
#       color: color to draw bounding box. Default is red.
#       thickness: line thickness. Default value is 4.
#       display_str_list: list of strings to display in box
#                         (each to be shown on its own line).
#       use_normalized_coordinates: If True (default), treat coordinates
#         ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
#         coordinates as absolute.
#     """
#     ymin, xmin, ymax, xmax = bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3]

#     draw = ImageDraw.Draw(image)
#     im_width, im_height = image.size
#     if use_normalized_coordinates:
#         (left, right, top, bottom) = (
#             xmin * im_width,
#             xmax * im_width,
#             ymin * im_height,
#             ymax * im_height,
#         )
#     else:
#         (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
#     if thickness > 0:
#         draw.line(
#             [(left, top), (left, bottom), (right, bottom), (right, top), (left, top)],
#             width=thickness,
#             fill=color,
#         )
#     try:
#         font = ImageFont.truetype("arial.ttf", 24)
#     except IOError:
#         font = ImageFont.load_default()

#     # If the total height of the display strings added to the top of the bounding
#     # box exceeds the top of the image, stack the strings below the bounding box
#     # instead of above.
#     display_str_heights = [
#         font.getbbox(ds)[3] - font.getbbox(ds)[1] for ds in display_str_list
#     ]
#     # Each display_str has a top and bottom margin of 0.05x.
#     total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

#     if top > total_display_str_height:
#         text_bottom = top
#     else:
#         text_bottom = bottom + total_display_str_height
#     # Reverse list and print from bottom to top.
#     for display_str in display_str_list[::-1]:
#         str_left, str_top, str_right, str_bottom = font.getbbox(display_str)
#         text_width, text_height = str_bottom - str_top, str_right - str_left
#         margin = np.ceil(0.05 * text_height)
#         draw.rectangle(
#             [
#                 (left, text_bottom - text_height - 2 * margin),
#                 (left + text_width, text_bottom),
#             ],
#             fill=color,
#         )
#         draw.text(
#             (left + margin, text_bottom - text_height - margin),
#             display_str,
#             fill="black",
#             font=font,
#         )
#         text_bottom -= text_height - 2 * margin



