require 'loadcaffe'
require 'cudnn'
require 'image'
require 'sys'

--torch.setdefaulttensortype('torch.DoubleTensor')

model = loadcaffe.load('CAM/models/deploy_alexnetplusCAM_imagenet.prototxt', 'CAM/models/alexnetplusCAM_imagenet.caffemodel', 'cudnn')
----model = loadcaffe.load('CAM/models/deploy_googlenetCAM.prototxt', 'CAM/models/imagenet_googleletCAM_train_iter_120000.caffemodel', 'cudnn')
----model = torch.load("alexnetplusCAM.torchmodel")
model:insert(nn.Squeeze(),22)
--model:remove(23)
--model:insert(nn.Linear(512,1))
--model:insert(nn.Sigmoid())
cudnn.convert(model, cudnn)
--model:cuda()
--print(model)

--torch.save("alexnetplusCAMCUDA.torchmodel", model)

table = {}
table[1] = {'bla'}
table[2] = {'ble'}
table2 = table:clone()
print(table2)


--data = image.load("CAM/dataset/Flickr32prepOriginal/classes/jpg/starbucks/4871375996.jpg", 3, "double")

---- convert to GPU
--cutorch.setDevice(1)
--data = data:cuda()
----data = torch.load('CAM/prepinput.dat')

--output = model:forward(data)

---- Extract and save softmax weights
--local mod = model.modules
--local weights_LR = mod[23].weight
--print(weights_LR:size())
--torch.save('weights_LR.dat',weights_LR)

---- Extract and save CAM output
--local cam = mod[13].output
--print(cam:size())
--torch.save('cam.dat', cam)

----print(output)
--print(output:size())
--torch.save('output.dat', output)
