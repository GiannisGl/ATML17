require 'loadcaffe'
require 'cudnn'
require 'image'
require 'nn'

model = loadcaffe.load('CAM/models/deploy_alexnetplusCAM_imagenet.prototxt', 'CAM/models/alexnetplusCAM_imagenet.caffemodel', 'cudnn')
--model = torch.load("alexnetplusCAMnn.torchmodel")

cudnn.convert(model, nn)
model:double()
model:remove(5)
model:insert(nn.SpatialConvolution(96, 256, 5,5,1,1,2,2),5)
model:remove(11)
model:insert(nn.SpatialConvolution(384,384,3,3,1,1,1,1),11)
model:remove(13)
model:insert(nn.SpatialConvolution(384,384,3,3,1,1,1,1),13)
model:remove(16)
model:insert(nn.SpatialConvolution(384,512,3,3,1,1,1,1),16)
model:remove(18)
model:insert(nn.SpatialConvolution(512,512,3,3,1,1,1,1),18)
model:remove(20)
model:insert(nn.SpatialAveragePooling(11,11,11,11),20)
model:insert(nn.Squeeze(), 22)
--model:remove(23)
--model:insert(nn.Linear(512,1))
model:insert(nn.Sigmoid())

model:double()
torch.save("alexnetplusCAMnn.torchmodel", model)
print(model)


