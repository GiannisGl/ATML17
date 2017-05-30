clear all;
source_folder = 'dataset/Flickr32separated/train/classes/jpg/';
dest_folder = 'dataset/Flickr32OriginalseparatedPrep/train/classes/jpg/';

dirs = dir(source_folder);
dirNames = {dirs([dirs.isdir]).name};
dirNames = dirNames(~ismember(dirNames,{'.','..'}));

for i=1:length(dirNames)
    subdir = cell2mat(dirNames(1,i));
    curdir = strcat(source_folder,subdir,'/')
    destdir = strcat(dest_folder,subdir,'/');
    mkdir(destdir);
    files = dir(curdir);
    fileNames = {files.name};
    fileNames = fileNames(~ismember(fileNames,{'.','..'}));
    for j=1:length(fileNames)
      name = cell2mat(fileNames(1,j));
      curfile = strcat(curdir, name)
      imgtmp = imread(curfile);
      if size(imgtmp,3) == 1
          imgtmp=imgtmp(:,:,[1 1 1]);
      end
      imgprep = prepare_image2(imgtmp);   
%      for l=1:size(imgprep,4)
%          destfile = strcat(destdir,num2str(l),'_', name);
%	  imgtowrite = imgprep(:,:,:,l);
%	  imwrite(imgtowrite, destfile,'jpg');     
%      end
      destfile = strcat(destdir,name);
      imwrite(imgprep,destfile,'jpg');
    end
end    


%input = imread('../dataset/Flickr32/classes/jpg/adidas/2399696288.jpg');
%imshow(input)
%size(input,3)
%imgtmp=input(:,:,[1 1 1]);
%size(imgtmp)
%imshow(imgtmp)
%input = imresize(input, [256 256]);
%prepinput = prepare_image(input);

%save('prepinput.mat','prepinput');
