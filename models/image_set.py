import os

from models.image import Image


class ImageSet:
    def __init__(self, root: str, load_image: bool, load_feather: bool, use_cuda: bool):
        # load image id list
        with open(os.path.join(root, 'list.txt')) as f:
            id_list = f.readlines()
        self.id_list = [x.replace('\n', '') for x in id_list]

        # set parameters
        self.root = root
        self.use_cuda = use_cuda
        self.load_image = load_image
        self.load_feather = load_feather

    def __getitem__(self, index: int):
        image = Image(self.root, self.id_list[index], self.load_image, self.load_feather, self.use_cuda)
        return image

    def __len__(self):
        return len(self.id_list)
