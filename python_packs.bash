#!/bin/bash

# this script adds pip and installs
# the python packages with their exact
# version that is on the car
sudo apt update && sudo apt upgrade
sudo apt install pip
sudo -H python -m pip install --upgrade 'pip<21.0'
sudo -H python -m pip install -r requirements.txt
