require 'image'
require 'lfs'
require 'nn'
torch.setdefaulttensortype('torch.DoubleTensor')

-- CONFIG
modelpath = 'alexnetplusCAMnn.torchmodel' 
inputfolderpath = 'CAM/dataset/Flickr32prepOriginal/classes/jpg/'
resultfolderpath = 'myresults'

-- CAM Creation helper functions
function create_CAM(lastconv, weights_LR)
	local lastconv_reshaped = torch.reshape(lastconv, lastconv:size(2)*lastconv:size(3), lastconv:size(1))
  local weights_LRtmp = weights_LR[{859,{}}]
	local detectionMap = torch.mv(lastconv_reshaped,weights_LRtmp)
	local cam = torch.reshape(detectionMap, lastconv:size(2), lastconv:size(3))
	return cam
end

function colourize_CAM(incam)
	local max = torch.max(incam)
  local min = torch.min(incam)
  local camnorm = torch.div(torch.add(incam,-min),max-min)
  local cam256 = torch.mul(camnorm, 256)
  local camceil = torch.ceil(cam256)
  camceil[camceil:lt(0.01)] = 0.01
	local outcam = image.y2jet(camceil)
  outcam = camnorm
	return outcam
end

function prepareCAM(data,model)
--  file = 'CAM/img2prep.jpg'
	mod = model.modules
  -- forward Data			
  img = data:clone()
  model:forward(data)

  -- Get CAM Data
  weights_LR = mod[23].weight:double()
  lastconv = mod[18].output:double()
  cam = create_CAM(lastconv, weights_LR)

  -- Create CAM
  cam = image.scale(cam, img:size(2), img:size(3))
  cam = colourize_CAM(cam)
--  threshold = 0.7
--  cam[cam:lt(threshold)]=0
--  cam[cam:ge(threshold)]=1
  cam = torch.repeatTensor(cam, 3,1,1)
  img_out = torch.add(torch.mul(img,0.3),torch.mul(cam,0.7))
  
  return img_out
  
end


function get_CAM_from_data(infolder, outfolder, modelpath)
	-- Load the model
	model = torch.load(modelpath)
	for dir in lfs.dir(infolder) do
		if dir ~= '.' and dir ~= '..' and dir=='starbucks' then
		  curdir = infolder.."/"..dir
		  print(curdir)
      i = 1
		  for file in lfs.dir(curdir) do
        if i>1 then break end
			if file ~= '.' and file ~= '..' then
        i = i+1
				curfile = curdir.."/"..file
				print(curfile)
        data = image.load(curfile, 3, "double")
--        data:mul(255):floor()
--        print(data)
				img_out = prepareCAM(data,model)
				image.save(outfolder.."/"..file,img_out)
			end
		  end
		end
	end
end

-- forward data
get_CAM_from_data(inputfolderpath,resultfolderpath,modelpath)
