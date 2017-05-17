input = imread('dataset/new/001_im.png');
imshow(input)
%input = imresize(input, [256 256]);
prepinput = prepare_image(input);

save('prepinput.mat','prepinput');