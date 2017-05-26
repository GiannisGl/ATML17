require 'lfs'
require 'image'


function prepare_dataset(folder)
    dataset = {}
    for dir in lfs.dir(folder) do
      if #dataset >= 20 then break end
      if dir ~= '.' and dir ~= '..' then
        curdir = folder.."/"..dir
        print(curdir)
        for file in lfs.dir(curdir) do
          if #dataset >= 20 then break end
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

--list= {}
--list[0] = 1
--list[1] = 2
--list[2] = 3
--list[3] = 3
--list[3] = 4
--list[5] = 2
--list[6] = 3
--list[4] = 2

--for i, v in ipairs(list) do
--    print(i .. " , " .. v)
--end
case = "train"
datasetfolder = "dataset/Flickr32seperated/"..case.."/classes/jpg"
dataset = prepare_dataset(datasetfolder)
dataSize = dataset:size()
print(dataSize)
torch.save( "datasetFlickr32_20.t7", dataset )


