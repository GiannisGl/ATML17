require 'torch'

local matio = require 'matio'

tensor = matio.load('prepinput.mat','prepinput')
print(tensor)

torch.save('prepinput.dat', tensor)
