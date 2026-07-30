import os
from .utils import Datum, DatasetBase

template = ['a photo of a {}, a type of walnut.']

class Walnut(DatasetBase):
    dataset_dir = 'Walnut_Color_Parvizi_3'

    def __init__(self, root, num_shots):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.template = template
        
        train = self.read_data(os.path.join(self.dataset_dir, str(num_shots)))
        test = self.read_data(os.path.join(self.dataset_dir, 'test'))

        super().__init__(train_x=train, val=test, test=test)
    
    def read_data(self, dir_path):
        items = []
        classes = sorted([d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))])
        label_map = {classname: i for i, classname in enumerate(classes)}
        
        for classname in classes:
            class_dir = os.path.join(dir_path, classname)
            label = label_map[classname]
            
            for img_name in os.listdir(class_dir):
                if not img_name.startswith('.'):
                    impath = os.path.join(class_dir, img_name)
                    item = Datum(
                        impath=impath,
                        label=label,
                        classname=classname
                    )
                    items.append(item)
        return items
