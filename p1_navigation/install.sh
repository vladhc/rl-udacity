#!/bin/bash

set -eu

wget https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip
unzip Banana_Linux_NoVis.zip
rm Banana_Linux_NoVis.zip

wget https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip
unzip Banana_Linux.zip
rm Banana_Linux.zip
