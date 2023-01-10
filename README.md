# IGN Object Detection

<p align="center">
<img src="assets/images/ign_logo.png?raw=true" width=20% height=20%>
</p>
<p align="center">
  <img src="assets/images/detection.png?raw=true" width=30% height=30%>
</p>


> *Disclaimer: this project is **not** affiliated with the IGN, it merely uses its API.*


## Project
This project aims to build a fully-fledged object detector using Tensorflow. It was built to :
- Improve my computer vision knowledge;
- Build a modern micro-service using an API;
- Work on EdgeML hardware : in my case, two Raspberry Pi 4B and a [Coral TPU accelerator](https://coral.ai/products/accelerator/);
- Enhance my software engineering level (documentation, GitHub actions, Docker...).

## Installation

### Docker image
A Docker image is available on Docker Hub. Install Docker, then run:

`docker pull louistransfer/object_detection_ign:latest`

To start the Starlite API, run:

`docker run -p 8000:8000 louistransfer/object_detection_ign:latest`

## Local install

Install conda or mamba as a package manager, then get the latest code from the repository by running:

`git clone https://github.com/louistransfer/object_detection_ign.git`

Then:

`cd object_detection_ign`

To install the environment, run:

`conda create -f environment.yml` then `conda activate satellite`

Then start the Starlite API by running:

`uvicorn main:app --reload`

## Usage

Once the API is running (check if the URL [http://localhost:8000/health](http://localhost:8000/health) returns the value *"healthy"*), you can perform inference on two endpoints.

There are 3 possible options to test such requests:

- Use the built-in swagger by accessing [http://localhost:8000/schema/swagger](http://localhost:8000/schema/swagger). By clicking on *"Try me out"*, you can fill an address or coordinates and receive an image as a response;
- Use the *"api_endpoints.ipynb"* notebook located in the **notebooks** folder, which contains pre-defined JSON payloads;
- Use curl to send a request to the API :

> On the address endpoint

`curl -X POST http://localhost:8000/inference/address -H "Content-Type: application/json" -d '{"address": "Trocadéro, Paris", "layer": "HR.ORTHOIMAGERY.ORTHOPHOTOS", "zoom_level": 19}' --output detection.jpg`

> On the location endpoint

`curl -X POST http://localhost:8000/inference/location -H "Content-Type: application/json" -d '{"latitude": 48.83980726885963,"longitude": -1.549046852─╯20273, "layer": "HR.ORTHOIMAGERY.ORTHOPHOTOS", "zoom_level": 19}' --output detection.jpg`

Both requests will be output to an image file called detection.jpg.


## Parameters description

**layer**: the set of images to use. The default is *"HR.ORTHOIMAGERY.ORTHOPHOTOS"*. See the list of layers on the IGN website [here](https://geoservices.ign.fr/services-web-experts-ortho).
**zoom_level**: level of zoom of the picture, from 1 (country scale) to 19 (neighborhood scale). In order for the detection model to work, it is recommended to stay at 19 (the maximum available zoom level).


###

## Architecture choices
The model is built on 3 independent modules :
- A WMTS Client which is used to either load satellite or aerial images from the IGN's WMTS API;
- An object detector leveraging [Tensorflow's Model Maker API](https://www.tensorflow.org/lite/models/modify/model_maker). The project was initially built on PyTorch, however the need to have a final model in .tflite format to use the Coral TPU required a framework change since conversions to the ONNX format failed;
- An API built using the [Starlite module](https://starlite-api.github.io/starlite/1.48/). FastAPI was considered, however the package's leadership is an [issue](https://github.com/tiangolo/fastapi/discussions/3970). Starlite is more robust and has managed to remove the Starlette dependency recently.