# ATML17


### To populate the models use
cd context-encoder 

bash ./models/scripts/download_inpaintCenter_models.sh

### To load caffee model
sudo apt-get install libprotobuf-dev protobuf-compiler

In OS X:
brew install protobuf
Then install the package itself:

luarocks install loadcaffe
In Ubuntu 16.04 you need to use gcc-5: CC=gcc-5 CXX=g++-5 luarocks install loadcaffe

Load a network:
require 'loadcaffe'
model = loadcaffe.load('deploy.prototxt', 'bvlc_alexnet.caffemodel', 'ccn2')
