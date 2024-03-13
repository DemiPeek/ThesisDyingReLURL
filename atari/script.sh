#!/bin/sh

rsync -av --exclude 'script.sh' --exclude 'wandb' --exclude 'networks' --exclude 'runs' --exclude 'venv' --exclude 'dqn_atari_original.py' ~/atari demi@145.108.195.26:~/demi

rsync -av --exclude 'script.sh' --exclude 'wandb' --exclude 'networks' --exclude 'runs' --exclude 'venv' --exclude 'dqn_atari_original.py' ~/atari demi@145.108.195.3:~/demi