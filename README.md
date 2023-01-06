# vqvae
Vector Quantisation Variational Auto Encoder

## Running on Nomagic's infrastructure

### Building docker image

```DOCKER_BUILDKIT=1 docker build . -f Dockerfile -t eu.gcr.io/gripper-ros/vqvae:latest```
```push eu.gcr.io/gripper-ros/vqvae:latest```
