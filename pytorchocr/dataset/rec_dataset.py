import os
import cv2
from torch.utils.data import Dataset
from torchvision import transforms

from pytorchocr.imgaug.transforms import create_transforms

class RecognitionDataset(Dataset):
    def __init__(self, config, mode, logger):
        self.logger = logger
        self.mode = mode.lower()

        global_config = config["Global"]
        dataset_config = config[mode]["dataset"]
        loader_config = config[mode]["loader"]

        self.data_dir = dataset_config['data_dir']
        self.image_dir = os.path.join(self.data_dir, self.mode, 'images')

        self.do_shuffle = loader_config['shuffle']
        self.annotations = self.load_annotations()
        self.transforms = self.load_transforms(dataset_config['transforms'])
        self.logger.info("Loaded the {} dataset. #samples = {}".format(self.mode, len(self.annotations)))

    def __len__(self):
        return len(self.annotations)


    def __getitem__(self, idx):
        image_file = self.annotations[idx]["image_file"]
        label = self.annotations[idx]["label"]

        image_path = os.path.join(self.image_dir, image_file)
        image = cv2.imread(image_path)

        if self.transforms:
            image = self.transforms(image)
            import ipdb; ipdb.set_trace()

        # TODO: convert label word into label index
        return { 'image' : image, 'label' : label }


    def load_annotations(self):
        gt_path = "{}/{}/gt.txt".format(self.data_dir, self.mode, 'gt.txt')
        image_dir = "{}/{}/images".format(self.data_dir, self.mode, 'images')

        assert os.path.exists(gt_path), "No such file: {}. Double check your path to gt file.".format(gt_path)
        assert os.path.exists(image_dir), "No such directory: {}. Double check your path to image_dir.".format(gt_path)

        annotations = []

        with open(gt_path, 'r') as f:
            for line in f.readlines():
                line = line.replace('\ufeff', '').strip()
                assert line.find(',') != 0, "line in gt.txt should be splitted by ',' delimiter."
                delimiter_ind = line.find(',')
                image_file = line[:delimiter_ind]
                label = line[delimiter_ind+1:].replace('"', '').strip()

                annotations.append({
                    'image_file' : image_file,
                    'label' : label
                })

        return annotations


    def load_transforms(self, transform_config):
        transform_funcs = create_transforms(transform_config)
        transform_composed = transforms.Compose(transform_funcs)
        return transform_composed
