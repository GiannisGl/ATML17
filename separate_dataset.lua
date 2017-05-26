require 'torch'

name = "Flickr32resizedFullAugmented"
datasetname = "dataset"..name..".t7"
dataset = torch.load(datasetname)
dataSize = dataset:size()

-- seperate datasets
seed = 1234
torch.manualSeed(seed)
shuffleIdx = torch.randperm(dataSize)
shuffleIdx2 = torch.randperm(dataSize)
shuffleIdx = shuffleIdx:index(1,shuffleIdx2:long())
trainpercent = 0.8
testpercent = 0.2
mintrainIdx = 1
maxtrainIdx = torch.floor(dataSize*trainpercent)
mintestIdx = maxtrainIdx+1
maxtestIdx = dataSize

-- create train dataset
trainset = {}
for i = 1,maxtrainIdx do
      table.insert(trainset,dataset[shuffleIdx[i]])
      end
function trainset:size() return #trainset end

--create test dataset
testset = {}
for i = mintestIdx,maxtestIdx do
      table.insert(testset,dataset[shuffleIdx[i]])
end
function testset:size() return #testset end
              
print("train dataset size: "..tostring(trainset:size()))
print("test dataset size: "..tostring(testset:size()))

torch.save( "dataset"..name.."train.t7", trainset )
torch.save( "dataset"..name.."test.t7", testset )


