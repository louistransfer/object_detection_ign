FROM mambaorg/micromamba:1.1.0
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml
USER root
ENV UVICORN_PORT=8000
EXPOSE ${UVICORN_PORT}
# Installs the Edge TPU driver
RUN apt-get update && apt-get install -y curl gnupg fontconfig && \ 
    echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | tee /etc/apt/sources.list.d/coral-edgetpu.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - && \ 
    apt-get update && apt-get install -y libedgetpu1-std && fc-cache -fv
# Installs micromamba, an extremely fast Conda alternative
RUN micromamba install -y -n base -f /tmp/environment.yml && \
    micromamba clean --all --yes 
ARG MAMBA_DOCKERFILE_ACTIVATE=1
WORKDIR /object_detection_ign/
COPY . /object_detection_ign/
CMD uvicorn main:app --reload --host 0.0.0.0