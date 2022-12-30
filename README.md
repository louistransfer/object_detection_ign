# IGN Object Detection
<!-- <div class="row">
  <div class="column">
    <img src="assets/images/detection.png?raw=true" width=30% height=30%>
  </div>
  <div class="column">
    
  </div> -->
<p align="center">
<img src="assets/images/ign_logo.png?raw=true" width=20% height=20%>
</p>
<p align="center">
  <img src="assets/images/detection.png?raw=true" width=30% height=30%>
</p>


<!-- ![IGN logo](assets/images/ign_logo.png?raw=true "Detection example" {width=480px height=480px}) -->


> *Disclaimer: this project is **not** affiliated with the IGN, it merely uses its API.*

<!-- ![Detection example](assets/images/detection.png?raw=true "Detection example" {width=270 height=138px}) -->


## Project
This project aims to build a fully-fledged object detector using Tensorflow. It was built to :
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