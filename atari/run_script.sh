#!/bin/sh

CUDA_VISIBLE_DEVICES=0 wandb agent demipeek/project_relu_demi/1y94wztx &

CUDA_VISIBLE_DEVICES=1  wandb agent demipeek/project_relu_demi/1y94wztx
