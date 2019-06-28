#!/bin/bash

set -eu

mkdir -p environments

if [ ! -f ./environments/Tennis_Linux/Tennis.x86_64 ]; then
  wget https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip
  unzip Tennis_Linux.zip
  rm Tennis_Linux.zip
  mv Tennis_Linux ./environments/Tennis_Linux
fi

if [ ! -f ./environments/Tennis_Linux_NoVis/Tennis.x86_64 ]; then
  wget https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip
  unzip Tennis_Linux_NoVis.zip
  rm Tennis_Linux_NoVis.zip
  mv Tennis_Linux_NoVis ./environments/Tennis_Linux_NoVis
fi

if [ ! -f ./environments/Crawler_Linux/Crawler.x86_64 ]; then
  wget https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Crawler/Crawler_Linux.zip
  unzip Crawler_Linux.zip
  rm Crawler_Linux.zip
  mv Crawler_Linux ./environments/Crawler_Linux
fi

if [ ! -f ./environments/Crawler_Linux_NoVis/Crawler.x86_64 ]; then
  wget https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Crawler/Crawler_Linux_NoVis.zip
  unzip Crawler_Linux_NoVis.zip
  rm Crawler_Linux_NoVis.zip
  mv Crawler_Linux_NoVis ./environments/Crawler_Linux_NoVis
fi

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
