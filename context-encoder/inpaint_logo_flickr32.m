function inpaint_logo_flickr32( imagepath )
% This function assumes...
% ... that the input image is from Flickr32
% ... the image path looks like '<path>/jpg/<brandname>/<imagename>.jpg
% ... and the corresponding bbox-file-path looks like
% <path>/masks/<brandname>/<imagename>.jpg.bboxes
% (This corresponds to the standard Flickr32 folder structure)

% Assemble bbox path
[pathstr,name,ext] = fileparts(imagepath);
parts = strsplit(pathstr,'/');
maskpath = '';
for i=1:length(parts)-2
  maskpath = [maskpath parts(i) '/'];
end
maskpath = deblank(strjoin([maskpath 'masks/' parts(end) '/' name ext '.bboxes.txt']));
maskpath(ismember(maskpath,' ')) = [];

inpaint_bbox(imagepath, maskpath)
end

