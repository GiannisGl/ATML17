require 'image'
require 'nn'
torch.setdefaulttensortype('torch.FloatTensor')

opt = {
    batchSize = 1,        -- number of samples to produce
    net = 'inpaintCenter/imagenet_inpaintCenter.t7',              -- path to the generator network
    imDir = 'images/new',            -- directory containing pred_center 
    name = 'imagenet_result',     -- name of the file saved
    gpu = 0,               -- gpu mode. 0 = CPU, 1 = 1st GPU etc.
    nc = 3,                -- # of channels in input
    manualSeed = 222,        -- 0 means random seed
    overlapPred = 4,       -- overlapping edges of center with context
    display = 1,
    fineSize = 128,        -- size of random crops
    nThreads = 1          -- # of data loading threads to use
}
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)

-- set seed
if opt.manualSeed == 0 then
    opt.manualSeed = torch.random(1, 10000)
end
print("Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)

-- load Context-Encoder
assert(opt.net ~= '', 'provide a generator model')
net = torch.load(opt.net)
net:apply(function(m) if m.weight then 
    m.gradWeight = m.weight:clone():zero(); 
    m.gradBias = m.bias:clone():zero(); end end)
net:evaluate()

-- initialize variables
inputSize = 128
image_ctx = torch.Tensor(opt.batchSize, opt.nc, inputSize, inputSize)
input_image_ctx = torch.Tensor(opt.batchSize, opt.nc, inputSize, inputSize)

-- port to GPU
if opt.gpu > 0 then
    require 'cunn'
    net:cuda()
    input_image_ctx = input_image_ctx:cuda()
else
   net:float()
end
print(net)

-- load data
for i=1,opt.batchSize do
    local imPath = string.format(opt.imDir.."/%03d_im.png",i)
    local input = image.load(imPath, nc, 'float')
    input = image.scale(input, inputSize, inputSize)
    input:mul(2):add(-1)
    image_ctx[i]:copy(input)
end
print('Loaded Image Block: ', image_ctx:size(1)..' x '..image_ctx:size(2) ..' x '..image_ctx:size(3)..' x '..image_ctx:size(4))


-- Generating random pattern
local res = 0.06 -- the lower it is, the more continuous the output will be. 0.01 is too small and 0.1 is too large
local density = 0.25
local MAX_SIZE = 10000
local low_pattern = torch.Tensor(res*MAX_SIZE, res*MAX_SIZE):uniform(0,1):mul(255)
local pattern = image.scale(low_pattern, MAX_SIZE, MAX_SIZE,'bicubic')
low_pattern = nil
pattern:div(255);
pattern = torch.lt(pattern,density):byte()  -- 25% 1s and 75% 0s
pattern = pattern:byte()
print('...Random pattern generated')

-- get random mask
local mask, wastedIter
wastedIter = 0
while true do
    local x = torch.uniform(1, MAX_SIZE-opt.fineSize)
    local y = torch.uniform(1, MAX_SIZE-opt.fineSize)
    mask = pattern[{{y,y+opt.fineSize-1},{x,x+opt.fineSize-1}}]  -- view, no allocation
    local area = mask:sum()*100./(opt.fineSize*opt.fineSize)
    if area>20 and area<30 then  -- want it to be approx 75% 0s and 25% 1s
        -- print('wasted tries: ',wastedIter)
        break
    end
    wastedIter = wastedIter + 1
end
mask=torch.repeatTensor(mask,opt.batchSize,1,1)
print(image_ctx:size())
print(mask:size())

-- original input image
real_center = image_ctx:clone() -- copy by value

-- fill masked region with mean value
image_ctx[{{},{1},{},{}}][mask] = 2*117.0/255.0 - 1.0
image_ctx[{{},{2},{},{}}][mask] = 2*104.0/255.0 - 1.0
image_ctx[{{},{3},{},{}}][mask] = 2*123.0/255.0 - 1.0
input_image_ctx:copy(image_ctx)

print("ctx size")
print(image_ctx:size())

-- run Context-Encoder to inpaint center
pred_center = net:forward(input_image_ctx)
print('Prediction: size: ', pred_center:size(1)..' x '..pred_center:size(2) ..' x '..pred_center:size(3)..' x '..pred_center:size(4))
print('Prediction: Min, Max, Mean, Stdv: ', pred_center:min(), pred_center:max(), pred_center:mean(), pred_center:std())

-- paste predicted region in the context
image_ctx[{{},{1},{},{}}][mask] = pred_center[{{},{1},{},{}}][mask]:float()
image_ctx[{{},{2},{},{}}][mask] = pred_center[{{},{2},{},{}}][mask]:float()
image_ctx[{{},{3},{},{}}][mask] = pred_center[{{},{3},{},{}}][mask]:float()

-- re-transform scale back to normal
input_image_ctx:add(1):mul(0.5)
image_ctx:add(1):mul(0.5)
pred_center:add(1):mul(0.5)
real_center:add(1):mul(0.5)

if opt.display then
    disp = require 'display'
    disp.image(pred_center, {win=1000, title=opt.name})
    -- disp.image(real_center, {win=1001, title=opt.name})
    disp.image(image_ctx, {win=1002, title=opt.name})
    print('Displayed image in browser !')
end


---- save outputs
-- image.save(opt.name .. '_predWithContext.png', image.toDisplayTensor(image_ctx))
-- image.save(opt.name .. '_realCenter.png', image.toDisplayTensor(real_center))
-- image.save(opt.name .. '_predCenter.png', image.toDisplayTensor(pred_center))

-- save outputs in a pretty manner
real_center=nil; pred_center=nil;
pretty_output = torch.Tensor(2*opt.batchSize, opt.nc, opt.fineSize, opt.fineSize)
input_image_ctx[{{},{1},{},{}}][mask] = 1.0
input_image_ctx[{{},{2},{},{}}][mask] = 1.0
input_image_ctx[{{},{3},{},{}}][mask] = 1.0
for i=1,opt.batchSize do
    pretty_output[2*i-1]:copy(input_image_ctx[i])
    pretty_output[2*i]:copy(image_ctx[i])
end
image.save(opt.name .. '.png', image.toDisplayTensor(pretty_output))
print('Saved predictions to: ./', opt.name .. '.png')
--image.display(image.toDisplayTensor(pretty_output))
