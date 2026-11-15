# Colab local runtime with the EvalRX notebook environment already installed.
#
# Hosted Colab (public or Google-internal) cannot start a user-supplied image.
# This image is for "Connect to a local runtime": run it on any machine with an
# NVIDIA GPU (workstation, cloud VM), then connect Colab to it. The notebooks
# find the source and environment at their default paths and skip setup.
#
# Build context: a checkout of this repo, or an unpacked offline bundle
# (tools/colab/build_bundle.sh; then the build needs no network):
#   docker build -f tools/colab/Dockerfile -t evalrx-colab:gpu .
#   docker run --gpus=all -p 127.0.0.1:9000:8080 evalrx-colab:gpu
# Colab: Connect > Connect to a local runtime > the printed
# http://127.0.0.1:9000/?token=... URL. Base image and port follow
# https://research.google.com/colaboratory/local-runtimes.html.
ARG BASE=us-docker.pkg.dev/colab-images/public/runtime

FROM ${BASE} AS build
ARG ACCEL=gpu
COPY . /content/evalrx
RUN EVALRX_ENV=/content/evalrx-env bash /content/evalrx/tools/colab/bootstrap.sh "${ACCEL}" \
    && rm -rf /content/evalrx/.bundle/wheels /content/evalrx/.bundle/python.tar.gz /content/evalrx/.bundle/uv

# Copy only the results: the bundle's wheels stay out of the final image.
FROM ${BASE}
COPY --from=build /content/evalrx /content/evalrx
COPY --from=build /content/evalrx-env /content/evalrx-env
