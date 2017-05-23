require 'loadcaffe'
require 'cudnn'
require 'image'
require 'nn'
--matio = require 'matio'

--model = loadcaffe.load('CAM/models/deploy_alexnetplusCAM_imagenet.prototxt', 'CAM/models/alexnetplusCAM_imagenet.caffemodel', 'cudnn')
model = torch.load("alexnetplusCAMnnFlickr.torchmodel")

--cudnn.convert(model, nn)
--model:double()
--model:remove(5)
--model:insert(nn.SpatialConvolution(96, 256, 5,5,1,1,2,2),5)
--model:remove(11)
--model:insert(nn.SpatialConvolution(384,384,3,3,1,1,1,1),11)
--model:remove(13)
--model:insert(nn.SpatialConvolution(384,384,3,3,1,1,1,1),13)
--model:remove(16)
--model:insert(nn.SpatialConvolution(384,512,3,3,1,1,1,1),16)
--model:remove(18)
--model:insert(nn.SpatialConvolution(512,512,3,3,1,1,1,1),18)
--model:remove(20)
--model:insert(nn.SpatialAveragePooling(11,11,11,11),20)
--model:insert(nn.Squeeze(), 22)
--model:remove(23)
--model:insert(nn.Linear(512,1))
--model:insert(nn.Sigmoid())

--torch.save("alexnetplusCAMnnFlickr.torchmodel", model)
--print(model)

-- load dataset
dataset = torch.load("datasetFlickr32.t7")
print("dataset loaded successfuly with size:")

-- create train and test datasets
seed = 1234
torch.manual_seed(seed)
dataSize = dataset:size()
print(dataSize)
shuffleIdx = torch.randperm(dataSize)
per = 0.7
trainIdx = torch.arange(1,dataSize*per)
trainIndices = shuffleIdx:index_select(trainIdx)
print(trainIndices:size())
--train_dataset = dataset:index_select(1,suffleIdx:long())
--print(train_dataset:size())




---- train the network
--criterion = nn.BCECriterion()
--trainer = nn.StochasticGradient(model, criterion)
--trainer.learningRate = 0.01
--trainer.maxIteration = 30
--trainer.verbose = False
--trainer.shuffleIndices = True
--trainer:train(train_dataset)



---- test accuracy
--threshold = 0.5
--correct = 0
--for i=1,n_test do
--    local groundtruth = test_labels[i]
--    local probability = model:forward(test_feats[i])
--    local prediction = probability:gt(threshold):double()
----     if i == 1 then
----         print("prob= ", probability[1])
----         print("prediction= ", prediction[1])
----         print("true label= ", groundtruth[1])
----     end
--    if groundtruth[1] == prediction[1] then
--        correct = correct + 1
--    end
--end

--accuracy = 100*correct/n_test

--print(correct,  accuracy.. ' % ')

