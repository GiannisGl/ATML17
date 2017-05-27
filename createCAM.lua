require 'cudnn'
require 'image'
require 'lfs'
torch.setdefaulttensortype('torch.DoubleTensor')

-- CONFIG
modelpath = 'modelFlickr32resizedFullAugmented.torchmodel' 
inputfolderpath = 'CAM/dataset/Flickr32prepOriginal/classes/jpg/'
resultfolderpath = 'results'

-- CAM Creation helper functions
function create_CAM(lastconv, weights_LR)
	lastconv_reshaped = torch.reshape(lastconv, lastconv:size(2)*lastconv:size(3), lastconv:size(1))
	detectionMap = torch.mv(lastconv_reshaped,weights_LR[{1,{}}])
	cam = torch.reshape(detectionMap, lastconv:size(2), lastconv:size(3))
	return cam
end

function colourize_CAM(cam)
	cam[cam:lt(0)] = 0.01
	local max = torch.max(cam)
	--cam = torch.div(cam,max)
	cam = image.y2jet(cam*1000)
	return cam
end

function get_CAM_from_data(infolder, outfolder, model)
	-- Load the model
	model = torch.load(modelpath)
	cudnn.convert(model, cudnn)
	model:cuda()
	mod = model.modules
	cutorch.setDevice(1)
	for dir in lfs.dir(infolder) do
		if dir ~= '.' and dir ~= '..' then
		  curdir = infolder.."/"..dir
		  print(curdir)
		  for file in lfs.dir(curdir) do
			if file ~= '.' and file ~= '..' then
				curfile = curdir.."/"..file
				print(curfile)
				
				-- forward Data			
				data = image.load(curfile, 3, "double")
				img = data:clone()
				data = data:cuda()
				model:forward(data)

				-- Get CAM Data
				weights_LR = mod[23].weight:double()
				lastconv = mod[18].output:double()
				cam = create_CAM(lastconv, weights_LR)

				-- Create CAM
				cam = image.scale(cam, img:size(2), img:size(3))
				cam = colourize_CAM(cam)
				img_out = torch.add(torch.mul(img,0.5),torch.mul(cam,0.5))
				--img_out = cam
				image.save(outfolder.."/"..file,img_out)
			end
		  end
		end
	end
end

-- forward data
get_CAM_from_data(inputfolderpath,resultfolderpath,modelpath)
