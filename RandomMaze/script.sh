#!/bin/sh

rsync -av --exclude 'backup_clusters.sh' --exclude 'wandb' --exclude 'venv' --exclude 'venv2' --exclude 'script.sh' ~/Code_Demi_Peek demi@145.108.195.26:~/demi