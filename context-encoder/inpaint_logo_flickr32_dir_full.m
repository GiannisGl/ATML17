clear all;

dirpath = "../Flickr32resultsBest/classes/jpg/"
dirs = dir(dirpath);
dirNames = {dirs([dirs.isdir]).name};
dirNames = dirNames(~ismember(dirNames,{'.','..'}));


for i=1:length(dirNames)
    subdir = cell2mat(dirNames(1,i));
    curdir = strcat(dirpath,subdir,'/');
    disp(curdir)
    files = dir(curdir);
    fileNames = {files.name};
    fileNames = fileNames(~ismember(fileNames,{'.','..'}));
    for j=1:length(fileNames)
      name = cell2mat(fileNames(1,j));
      curfile = strcat(curdir, name);
      disp(curfile)
      
%       imagepath = deblank(strjoin(curfile));
        imagepath = curfile;
        % Assemble bbox path
        [pathstr,name,ext] = fileparts(imagepath);
        parts = strsplit(pathstr,'/');
        maskpath = '';
        for i=1:length(parts)-2
          maskpath = [maskpath parts(i) '/'];
        end
        bboxpath = deblank(strjoin([maskpath 'masks/' parts(end) '/' name ext '.bboxes.txt']));
        bboxpath(ismember(bboxpath,' ')) = [];
        try
          inpaint_bbox(imagepath, bboxpath)
        catch
        disp('err')
        end_try_catch
      
      
%      break;
    end
%    break;
end    
