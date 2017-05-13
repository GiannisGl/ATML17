require 'loadcaffe'
require 'cudnn'
require 'image'
require 'itorch'

model = loadcaffe.load('CAM/models/deploy_alexnetplusCAM_imagenet.prototxt', 'CAM/models/alexnetplusCAM_imagenet.caffemodel', 'cudnn')
print(model)

input = image.load("CAM/dataset/new/001_im_prep.png", 3, "double")

--output = model:forward(input)

--image.display(input)
