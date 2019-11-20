# Meetup Data Toulouse: Experiment Tracking With Mlflow

This repo is an adaptation of the official *pytorch* examples of mnist training.  
Upon this repository we add a layer for tracking hyperparameters and metrics with
a tool called mlflow.  
This is part of a presentation of this tool in a meetup in Toulouse in 2019.

To reproduce the experiment, just run the `train_mniyst.sh`. which will run 5 times the training with different hyperparamters. Don't forget to install the dependencies (torch ... etc).

To start the mlflow dashboard, just run `mlflow server` where the mlruns folder resides.


Author: [Hacene Karrad](https://www.linkedin.com/in/hacene-karrad-66893a159/), Computer Vision Engineer at [Delair](https://delair.aero).