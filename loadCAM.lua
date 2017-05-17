require 'loadcaffe'
require 'cudnn'
require 'image'
require 'cutorch'
require 'nn'
--matio = require 'matio'

model = loadcaffe.load('CAM/models/deploy_alexnetplusCAM_imagenet.prototxt', 'CAM/models/alexnetplusCAM_imagenet.caffemodel', 'cudnn')
--model = loadcaffe.load('CAM/models/deploy_googlenetCAM.prototxt', 'CAM/models/imagenet_googleletCAM_train_iter_120000.caffemodel', 'cudnn')
--model = torch.load("alexnetplusCAM.torchmodel")
print(model)
--model:evaluate()
--torch.save("alexnetplusCAMnn.torchmodel", model)

input = image.load("CAM/dataset/new/001_im_prep.png", 3, "double")
--input = matio.load("CAM/dataset/new/002_im_prep.mat", "prepinput")

--image.display(input)
print(input:size())

--input:cuda()
--print(input)

--output = model:forward(input)
--print(output:size())
--print(output)
