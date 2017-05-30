require 'torch'
require 'image'
require 'lfs'
require 'nn'
require 'cudnn'
torch.setdefaulttensortype('torch.DoubleTensor')

-- CONFIGfd
modelpath = "modelFlickr32OriginalseparatedPrep1.torchmodel"
inputfolderpath = 'CAM/dataset/Flickr32OriginalseparatedPrep/test/classes/jpg/'
originalfolderpath = "CAM/dataset/Flickr32separated/test/classes/jpg/"
outputfolderpath = "CAM/dataset/Flickr32results/classes/"
ourmasksfolderpath = outputfolderpath.."masks/"
resultsfolderpath = outputfolderpath.."CAMresults/"
croppedfolderpath = outputfolderpath.."jpg/"

-- CAM Creation helper functions
function create_CAM(lastconv, weights_LR)
	-- reshape in torch write rowwise
	local lastconv_reshaped = torch.reshape(lastconv,lastconv:size(1), lastconv:size(2)*lastconv:size(3))
        lastconv_reshaped = lastconv_reshaped:t()
        local weights_LRtmp = weights_LR[{1,{}}]
        local detectionMap = torch.mv(lastconv_reshaped,weights_LRtmp)
        local cam = torch.reshape(detectionMap, lastconv:size(3), lastconv:size(2))
        return cam:t()

end

function colourize_CAM(incam)
	local cam256 = torch.mul(incam, 256)
	local camceil = torch.ceil(cam256)
	camceil[camceil:lt(0.01)] = 0.01
	local outcam = image.y2jet(camceil)

	return outcam
end

function prepareCAM(data,model)
--  file = 'CAM/img2prep.jpg'
	mod = model.modules
  -- forward Data			
  img = data:clone()
  data = data:cuda()
  model:forward(data)

  -- Get CAM Data
  weights_LR = mod[23].weight:double()
  lastconv = mod[18].output:double()

  -- Create CAM
  cam = create_CAM(lastconv, weights_LR)
  cam = normalizeIMG(cam)
  cam = image.scale(cam, img:size(2), img:size(3))
  
  return cam
  
end

function thresholdIMG(img, threshold)
	img_out = img:clone()
	img_out[img_out:lt(threshold)]=0
	img_out[img_out:ge(threshold)]=1

	return img_out
end

function normalizeIMG(img)
	local max = torch.max(img)
	local min = torch.min(img)
	local imgnorm = torch.div(torch.add(img,-min),max-min)
  
	return imgnorm
end

function combineCAMwithIMG(cam, img)
	cam = colourize_CAM(cam)
	img = image.scale(img, cam:size(2), cam:size(3), 'bilinear')
	img_out = torch.add(torch.mul(img,0.3),torch.mul(cam,0.7))

	return img_out
end


function get_CAM_from_data(infolder, outfolder, masksfolder, originalfolder, croppedfolder, modelpath)
	-- Load the model
	model = torch.load(modelpath)
	cudnn.convert(model, cudnn)
        model:cuda()
        cutorch.setDevice(1)
	for dir in lfs.dir(infolder) do
		if dir ~= '.' and dir ~= '..' and dir ~='no-logo' then
		  curdir = infolder..dir
		  print(curdir)
	     	  i = 1
		  for file in lfs.dir(curdir) do
--		        if i>5 then break end
			if file ~= '.' and file ~= '..' then
			        i = i+1
				curfile = curdir.."/"..file
				print(curfile)
			        img = image.load(curfile, 3, "double")
				originalfile = originalfolder..dir.."/"..file
				print("original "..originalfile)	
				originalimg = image.load(originalfile)
				originalimg = image.scale(originalimg, img:size(2), img:size(3), 'bilinear')
				cam = prepareCAM(img, model)
				mask = thresholdIMG(cam, 0.7)
				
				lfs.mkdir(masksfolder..dir)
				maskfile = masksfolder..dir.."/"..file
				print("mask "..maskfile)
				image.save(maskfile, mask)

				result = combineCAMwithIMG(cam,originalimg)
				lfs.mkdir(outfolder..dir)
				resultfile = outfolder..dir.."/"..file
				print("CAMresult "..resultfile)	
				image.save(resultfile,result)
				
				croppedfile = croppedfolder..dir.."/"..file
				lfs.mkdir(croppedfolder..dir)
				print("cropped "..croppedfile)
				mask = torch.repeatTensor(mask, 3,1,1)		
				croppedIMG = torch.cmul(originalimg, mask)
				image.save(croppedfile, croppedIMG)
			end
		  end
		end
	end
end

-- forward data
get_CAM_from_data(inputfolderpath,resultsfolderpath, ourmasksfolderpath,originalfolderpath, croppedfolderpath, modelpath)
