  
create_sub_dataset('CAM/dataset/Flickr32prepOriginal/', 'CAM/dataset/Flickr32OriginalseparatedPrep/test/', 'trainset.relpaths.txt')

create_sub_dataset('CAM/dataset/Flickr32prepOriginal/', 'CAM/dataset/Flickr32OriginalseparatedPrep/train/', 'valset.relpaths.txt')

create_sub_dataset('CAM/dataset/Flickr32prepOriginal/', 'CAM/dataset/Flickr32OriginalseparatedPrep/val/', 'testset.relpaths.txt')

%data_augmentation('CAM/dataset/Flickr32separated/train/')
%data_augmentation('CAM/dataset/Flickr32separated/val/')
%data_augmentation('CAM/dataset/Flickr32separated/test/')

