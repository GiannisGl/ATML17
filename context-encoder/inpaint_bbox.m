function inpaint_bbox (imagepath, bboxpath)
pkg load image
% inpaints the image using the context encoder on the given bbox
% You need the 'image' package for octave: type "pkg install -forge image" in an octave shell (not in unix shell)

% Get image name
[path,name,ext] = fileparts(imagepath);
% Target image dimension for the network input
DIM = 128;

% get bbox coordinates from file
% x y width height (x from left, y from top)
box_coords = dlmread(bboxpath, SEP=' ', R0=1, C0=0);
box_x = box_coords(1);
box_y = box_coords(2);
box_w = box_coords(3);
box_h = box_coords(4);
% as the bbox is made square we have to shift the beginning coordinates accordingly
box_size = 2*round(max(box_w, box_h)/2);
wh_diff = box_coords(3)-box_coords(4);
x_shift = ceil(min(wh_diff/2,0));
y_shift = -ceil(max(wh_diff/2,0));
box_x = box_x+x_shift;
box_y = box_y+y_shift;

% the image has to be 128x128 and the box 64x64
out_img_size = 2*box_size;
ratio = (DIM)/out_img_size;

% load image
img = imread(imagepath);
img_w = size(img,2);
img_h = size(img,1);
% get part with logo in center (but limited with image boundaries, if the logo is on the edge)
%img_part_xl = max(box_x-0.5*box_size,0); % for matlab min is 0
 mg_part_xl = max(box_x-0.5*box_size,1);
img_part_xr = min(box_x+1.5*box_size,img_w);
%img_part_yt = max(box_y-0.5*box_size,0); % for matlab min is 0
img_part_yt = max(box_y-0.5*box_size,1);
img_part_yb = min(box_y+1.5*box_size,img_h);
img_part = img(img_part_yt:img_part_yb, img_part_xl:img_part_xr, :);

% resize the image
net_img = imresize(img_part,[128 128]);
% save it in the appropriate location so the lua script finds it
imwrite(net_img,'images/001_im.png');

% the removing of the center is done by the lua script
system("th context_encoder_run.lua");

% Get the result and paste it back into the image
<<<<<<< HEAD
imfilled = imread('filled_logo_predWithContext.png');
imfilled = imresize(imfilled,[size(img_part,1),size(img_part,2)]);
img(img_part_yt:img_part_yb, img_part_xl:img_part_xr, :) = imfilled;
imwrite(img,'finished_product2.png');
=======
imfilled = imread('filled_logo_predCenter.png');
centersize_y = (img_part_yb-box_size/2)-(img_part_yt+box_size/2);
centersize_x = (img_part_xr-box_size/2)-(img_part_xl+box_size/2);
imfilled = imresize(imfilled,[centersize_x,centersize_y]);
img_removed = img;
img_removed(img_part_yt+box_size/2:img_part_yb-box_size/2-1, img_part_xl+box_size/2:img_part_xr-box_size/2-1, :) = imfilled;
imwrite(img_removed,['results/' name '_removed.png']);
imwrite(img,['results/' name '.png']);
>>>>>>> 3974f46906a00a77f68854c3b292fafb940e2e38

endfunction
