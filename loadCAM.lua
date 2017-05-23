require 'loadcaffe'
require 'cudnn'
require 'image'

torch.setdefaulttensortype('torch.DoubleTensor')

model = loadcaffe.load('CAM/models/deploy_alexnetplusCAM_imagenet.prototxt', 'CAM/models/alexnetplusCAM_imagenet.caffemodel', 'cudnn')
--model = loadcaffe.load('CAM/models/deploy_googlenetCAM.prototxt', 'CAM/models/imagenet_googleletCAM_train_iter_120000.caffemodel', 'cudnn')
--model = torch.load("alexnetplusCAM.torchmodel")
model:insert(nn.Squeeze(),22)
model:cuda()
print(model)

--torch.save("alexnetplusCAM.torchmodel", model)

-- They don't have hdf5 installed
--local myFile = hdf5.open('CAM/prepinput.h5', 'r')
--local data = myFile:read('/data'):all()
--myFile.close()

data = image.load("CAM/dataset/new/001_im_prep.png", 3, "double")

-- convert to GPU
cutorch.setDevice(1)
data = data:cuda()
--data = torch.load('CAM/prepinput.dat')

output = model:forward(data)
print(output)
