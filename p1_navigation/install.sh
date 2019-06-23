#!/bin/bash

set -eu

if [ ! -f ./environments/Reacher_Linux_NoVis/Reacher.x86_64 ]; then
  wget https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip
  unzip Reacher_Linux_NoVis.zip
  rm Reacher_Linux_NoVis.zip
  mv Reacher_Linux_NoVis ./environments/Reacher_Linux_NoVis
fi

if [ ! -f ./environments/Reacher_Linux_NoVis_single/Reacher.x86_64 ]; then
  wget https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip
  unzip Reacher_Linux_NoVis.zip
  rm Reacher_Linux_NoVis.zip
  mv Reacher_Linux_NoVis ./environments/Reacher_Linux_NoVis_single
fi

if [ ! -f ./environments/Reacher_Linux_single/Reacher.x86_64 ]; then
  wget https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip
  unzip Reacher_Linux.zip
  rm Reacher_Linux.zip
  mv Reacher_Linux ./environments/Reacher_Linux_single
fi

if [ ! -f ./environments/Reacher_Linux/Reacher.x86_64 ]; then
  wget https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip
  unzip Reacher_Linux.zip
  rm Reacher_Linux.zip
  mv Reacher_Linux ./environments/
fi

if [ ! -f ./environments/Banana_Linux_NoVis/Banana.x86_64 ]; then
  wget https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip
  unzip Banana_Linux_NoVis.zip
  rm Banana_Linux_NoVis.zip
  mv Banana_Linux_NoVis ./environments/
fi

if [ ! -f ./environments/Banana_Linux/Banana.x86_64 ]; then
  wget https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip
  unzip Banana_Linux.zip
  rm Banana_Linux.zip
  mv Banana_Linux ./environments/
fi
