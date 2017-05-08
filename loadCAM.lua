require 'loadcaffe'
require 'nn'
require 'cudnn'
require 'image'
disp = require 'display'

model = loadcaffe.load('CAM/models/deploy_alexnetplusCAM_imagenet.prototxt', 'CAM/models/imagenet_googleletCAM_train_iter_120000.caffemodel', 'cudnn')
--print(model)

input = image.load("CAM/dataset/new/001_im_prep.png", 3, "double")
disp(input)

model:forward(input)
