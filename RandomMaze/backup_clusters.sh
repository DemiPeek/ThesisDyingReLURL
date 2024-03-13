#!/bin/sh


rsync -av --exclude 'runs' --exclude 'wandb' --exclude 'saved_runs' --exclude 'venv' --exclude 'new' --exclude 'backup_clusters.sh' --exclude 'download_run_files.sh'  ~/PycharmProjects/Code_Demi_Peek jacob@145.108.195.3:~/jacob

rsync -av --exclude 'runs' --exclude 'wandb' --exclude 'saved_runs' --exclude 'venv' --exclude 'new' --exclude 'backup_clusters.sh' --exclude 'download_run_files.sh' ~/PycharmProjects/Code_Demi_Peek jacob@145.108.195.26:~/jacob

