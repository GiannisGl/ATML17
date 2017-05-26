  
create_sub_dataset('CAM/dataset/Flickr32/', 'CAM/dataset/Flickr32separated/test/', 'trainset.relpaths.txt')

create_sub_dataset('CAM/dataset/Flickr32/', 'CAM/dataset/Flickr32separated/train/', 'valset.relpaths.txt')

create_sub_dataset('CAM/dataset/Flickr32/', 'CAM/dataset/Flickr32separated/val/', 'testset.relpaths.txt')

data_augmentation('CAM/dataset/Flickr32separated/train/')
data_augmentation('CAM/dataset/Flickr32separated/val/')
data_augmentation('CAM/dataset/Flickr32separated/test/')