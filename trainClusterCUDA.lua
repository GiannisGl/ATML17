require 'loadcaffe'
require 'cudnn'
require 'image'
require 'nn'
require 'optim'
require 'sys'

torch.setdefaulttensortype('torch.DoubleTensor')

--model = loadcaffe.load('CAM/models/deploy_alexnetplusCAM_imagenet.prototxt', 'CAM/models/alexnetplusCAM_imagenet.caffemodel', 'cudnn')
model = torch.load("alexnetplusCAMFlickr32CUDA.torchmodel")
model:insert(nn.Squeeze(),22)
model:remove(23)
model:insert(nn.Linear(512,1))
model:insert(nn.Sigmoid())
cudnn.convert(model, cudnn)
model:cuda()

torch.save("alexnetplusCAMFlickr32CUDA.torchmodel", model)
print("new model for Flickr32 binary")
print(model)

-- load dataset
name = 'Flickr32resizedOriginal'
datasetfilename = 'dataset'..name..'.t7'
print(datasetfilename)
dataset = torch.load(datasetfilename)
print("dataset loaded successfuly with size:")
dataSize = dataset:size()
--dataSize = torch.Tensor(dataSize)
print("dataset size: "..tostring(dataset:size()))
print(torch.type(dataSize))

-- create datasets
seed = 1234
torch.manualSeed(seed)
shuffleIdx = torch.randperm(dataSize)
trainpercent = 0.6
valpercent = 0.2
testpercent = 0.2
mintrainIdx = 1
maxtrainIdx = torch.floor(dataSize*trainpercent)
minvalIdx = maxtrainIdx+1
maxvalIdx = maxtrainIdx+torch.floor(dataSize*valpercent)
mintestIdx = maxvalIdx+1
maxtestIdx = dataSize

-- create train dataset
trainset = {}
for i = 1,maxtrainIdx do
      table.insert(trainset,dataset[shuffleIdx[i]])
end
function trainset:size() return #trainset end

-- create val dataset
valset = {}
for i = minvalIdx,maxvalIdx do
      table.insert(valset,dataset[shuffleIdx[i]])
end
function valset:size() return #valset end

--create test dataset
testset = {}
for i = mintestIdx,maxtestIdx do
      table.insert(testset,dataset[shuffleIdx[i]])
end
function testset:size() return #testset end
  
print("train dataset size: "..tostring(trainset:size()))
print("val dataset size: "..tostring(valset:size()))
print("test dataset size: "..tostring(testset:size()))

imgsize = dataset[1][1]:size()
channels = imgsize[1]
imgwidth = imgsize[2]
imgheight = imgsize[3]


-- choose criterion
criterion = nn.BCECriterion()
criterion:cuda()

-- define the descent algorithm
sgd_params = {
   learningRate = 1e-2,
   learningRateDecay = 1e-4,
   weightDecay = 1e-3,
   momentum = 1e-4
}

x, dl_dx = model:getParameters()

-- step function
step = function(batch_size)
    local current_loss = 0
    local shuffle = torch.randperm(trainset:size())
    batch_size = batch_size or 32

    for t = 1,trainset:size(),batch_size do
        -- setup inputs for this mini-batch
        -- no need to setup targets, since they are the same
        local size = math.min(t + batch_size - 1, trainset:size()) - t
        local inputs = torch.Tensor(size, channels, imgwidth, imgheight)
        local targets = torch.Tensor(size)
        for i = 1,size do
            inputs[i] = torch.Tensor(trainset[shuffle[i+t]][1])
        end
        for i = 1,size do
            targets[i] = torch.Tensor(trainset[shuffle[i+t]][2])
        end
        inputs = inputs:cuda()
        targets = targets:cuda()

        local feval = function(x_new)
            -- reset data
            if x ~= x_new then x:copy(x_new) end
            dl_dx:zero()

            -- perform mini-batch gradient descent
            local loss = criterion:forward(model:forward(inputs), targets)
            model:backward(inputs, criterion:backward(model.output, targets))

            return loss, dl_dx
        end
        
         _, fs = optim.sgd(feval, x, sgd_params)
        -- fs is a table containing value of the loss function
        -- (just 1 value for the SGD optimization)
        current_loss = current_loss + fs[1]
    end

    return current_loss
end

-- evaluation function on a seperate dataset
eval = function(dataset, batch_size)
    local loss = 0
    batch_size = batch_size or 32

    for t = 1,dataset:size(),batch_size do
        local size = math.min(t + batch_size - 1, dataset:size()) - t
        local inputs = torch.Tensor(batch_size, channels, imgwidth, imgheight)
        local targets = torch.Tensor(batch_size)
        for i = 1,size do
            inputs[i] = torch.Tensor(dataset[i+t][1])
        end
        for i = 1,size do
            targets[i] = torch.Tensor(dataset[i+t][2])
        end
        inputs = inputs:cuda()
        targets = targets:cuda()
        local outputs = model:forward(inputs)
        loss = loss + criterion:forward(model:forward(inputs), targets)
    end

    return loss
end

max_iters = 30

-- train the model
print("started training")
model_last = model
do
    local last_loss = 0
    local increasing = 0
    local threshold = 1 -- how many increasing epochs we allow
    for i = 1,max_iters do
        sys.tic()
        model_last = model
        local loss = step()
        print(string.format('Epoch: %d Current loss: %4f', i, loss))
        local validation_loss = eval(valset)
        print(string.format('Loss on the validation set: %4f', validation_loss))
        if last_loss < validation_loss then
            if increasing > threshold then break end
            increasing = increasing + 1
        else
            increasing = 0
        end
        last_loss = validation_loss
        print(string.format('elapsed time: %f seconds', sys.toc()))
    end
end

-- test accuracy on the test dataset
test_loss = eval(testset)
print(string.format('Loss on the test set: %4f', test_loss))

modelfilename = 'model'..name..'.torchmodel'
torch.save(modelfilename,model_last)

-- test accuracy
threshold = 0.5
correct = 0
for i=1,n_test do
    local groundtruth = test_labels[i]
   local probability = model:forward(test_feats[i])
   local prediction = probability:gt(threshold):double()
     if i == 1 then
         print("prob= ", probability[1])
         print("prediction= ", prediction[1])
         print("true label= ", groundtruth[1])
     end
    if groundtruth[1] == prediction[1] then
        correct = correct + 1
    end
end

accuracy = 100*correct/n_test

print(correct,  accuracy.. ' % ')






