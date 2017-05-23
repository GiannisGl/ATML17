
local matio = require 'matio'

weights_LR = torch.load('weights_LR.dat','ascii')
cam = torch.load('cam.dat','ascii')
output = torch.load('output.dat','ascii')

matio.save('cam_outputs.mat', {weights_LR=weights_LR, cam=cam, scores=output})

