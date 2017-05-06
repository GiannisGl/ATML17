require 'loadcaffe'

model = loadcaffe.load('deploy_alexnetplusCAM_imagenet.prototxt', 'bvlc_alexnet.caffemodel', 'ccn2')
print(model)