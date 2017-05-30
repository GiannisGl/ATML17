clear all;

dirpath = "../Flickr32resultsBest/classes/masks/"
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
      imgtmp = double(imread(curfile));
      if size(imgtmp,3) > 1
          imgtmp=imgtmp(:,:,1);
      end
      [rows, cols] = find(imgtmp==1);
      x = min(cols);
      xmax = max(cols);
      width = floor((xmax-x)/2);
      x = x+floor(width/2);
      y = min(rows);
      ymax = max(rows);
      height = floor((ymax-y)/2);
      y = y+floor(height/2);
      
      % save bbox
      [path,nametmp,ext] = fileparts(curfile);
      bboxfile = strcat(curdir,nametmp,ext,'.bboxes.txt');
      fid = fopen(bboxfile, 'wt' );
      header = 'x y width height\n';
      data = [x, y, width, height];
%      data = [y, x, height, width];
      fprintf(fid,  header);
      fprintf(fid,  '%d %d %d %d\n', data);
      
      fclose(fid);
      
      
      
%      break;
    end
%    break;
end    
