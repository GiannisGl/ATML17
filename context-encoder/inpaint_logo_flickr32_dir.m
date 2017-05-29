function inpaint_logo_flickr32_dir(dirpath)

imagepaths = glob([dirpath '/*.jpg']);

for i = 1:length(imagepaths)
  imagepath = deblank(strjoin(imagepaths(i)));

  % Assemble bbox path
  [pathstr,name,ext] = fileparts(imagepath);
  parts = strsplit(pathstr,'/');
  maskpath = '';
  for i=1:length(parts)-2
    maskpath = [maskpath parts(i) '/'];
  end
  maskpath = deblank(strjoin([maskpath 'masks/' parts(end) '/' name ext '.bboxes.txt']));
  maskpath(ismember(maskpath,' ')) = [];
  try
    inpaint_bbox(imagepath, maskpath)
  catch
    %do nothing
  end_try_catch
end
end