require 'loadcaffe'
require 'cudnn'
require 'image'

--model = loadcaffe.load('CAM/models/deploy_alexnetplusCAM_imagenet.prototxt', 'CAM/models/alexnetplusCAM_imagenet.caffemodel', 'cudnn')
model = torch.load("alexnetplusCAM.torchmodel")
print(model)

torch.save("alexnetplusCAM.torchmodel", model)

--input = image.load("CAM/dataset/new/001_im_prep.png", 3, "double")

--output = model:forward(input)
--print(output)
