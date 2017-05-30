require 'lfs'
require 'image'


function prepare_dataset(folder)
    print(folder)
    dataset = {}
    for dir in lfs.dir(folder) do
      --if #dataset >= 20 then break end
      if dir ~= '.' and dir ~= '..' then
        curdir = folder.."/"..dir
        print(curdir)
        for file in lfs.dir(curdir) do
        --if #dataset >= 20 then break end
          if file ~= '.' and file ~= '..' then
            curfile = curdir.."/"..file
            print(curfile)
            label = torch.Tensor(1)
            label[1] = 1
            if string.find(dir, 'no-logo') then
              label[1] = 0
            end
            imgtmp = image.load(curfile, 3, "double")
            dataset[#dataset+1] = {imgtmp, label}
          end
        end
      end
    end
    
    function dataset:size() return #dataset end 
   
   return dataset
end


case = "train"
print("train")
datasetfolder = "CAM/dataset/Flickr32OriginalseparatedPrep/"..case.."/classes/jpg"
dataset = prepare_dataset(datasetfolder)
dataSize = dataset:size()
print(dataSize)
torch.save( "datasetFlickr32OriginalseparatedPrep"..case..".t7", dataset )


case = "val"
print("val")
datasetfolder = "CAM/dataset/Flickr32OriginalseparatedPrep/"..case.."/classes/jpg"
dataset = prepare_dataset(datasetfolder)
dataSize = dataset:size()
print(dataSize)
torch.save( "datasetFlickr32OriginalseparatedPrep"..case..".t7", dataset )


case = "test"
print("test")
datasetfolder = "CAM/dataset/Flickr32OriginalseparatedPrep/"..case.."/classes/jpg"
dataset = prepare_dataset(datasetfolder)
dataSize = dataset:size()
print(dataSize)
torch.save( "datasetFlickr32OrginalseparatedPrep"..case..".t7", dataset )


