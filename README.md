# IGN Object Detection
![Detection example](assets/images/detection.png?raw=true "Detection example")

## Project
This project aims to build a fully-fledged object detector using Tensorflow. I built it to :
- Improve my computer vision knowledge;
- Build a modern micro-service using an API;
- Work on EdgeML hardware : in my case, two Raspberry Pi 4B and a [Coral TPU accelerator](https://coral.ai/products/accelerator/);
- Enhance my software engineering level (documentation, GitHub actions, Docker...).

## Installation
WIP.

## Architecture choices
The model is built on 3 independent modules :
- A WMTS Client which is used to load satellite images from IGN's WMTS API;
- An object detector leveraging [Tensorflow's Model Maker API](https://www.tensorflow.org/lite/models/modify/model_maker). The project was initially built on PyTorch, however the need to have a final model in .tflite format to use the Coral TPU required a framework change since conversions to the ONNX format failed;
- An API built using the [Starlite module](https://starlite-api.github.io/starlite/1.48/). FastAPI was considered, however the package's leadership is an [issue](https://github.com/tiangolo/fastapi/discussions/3970). Starlite is more robust and has managed to remove the Starlette dependency recently.