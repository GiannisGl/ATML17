require 'loadcaffe'
require 'cudnn'
require 'image'

torch.setdefaulttensortype('torch.DoubleTensor')

--model = loadcaffe.load('CAM/models/deploy_alexnetplusCAM_imagenet.prototxt', 'CAM/models/alexnetplusCAM_imagenet.caffemodel', 'cudnn')
--model = loadcaffe.load('CAM/models/deploy_googlenetCAM.prototxt', 'CAM/models/imagenet_googleletCAM_train_iter_120000.caffemodel', 'cudnn')
model = torch.load("alexnetplusCAM.torchmodel")
model:insert(nn.Squeeze(),22)
model:remove(23)
model:insert(nn.Linear(512,1))
model:insert(nn.Sigmoid())
cudnn.convert(model, cudnn)
model:cuda()
print(model)
--local mod = model.modules

torch.save("alexnetplusCAMFlickr32.torchmodel", model)


