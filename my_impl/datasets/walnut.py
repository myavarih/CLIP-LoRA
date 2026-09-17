import os
import numpy as np
import torch
from .utils import Datum, DatasetBase

template = ['a photo of a {} walnut, a type of walnut.']

CANONICAL_CLASSES = [
    'siah goshti',
    'brown plus',
    'brown momtaz',
    'white mamooli',
    'white momtaz',
    'lux'
]

CLASS_MAP = {
    'brown momtaz': 'premium brown',
    'brown plus': 'high-quality brown',
    'lux': 'luxury extra light',
    'siah goshti': 'dark meaty',
    'white mamooli': 'standard white',
    'white momtaz': 'premium white'
}

# Semantic / partial ordinal cost matrix C in R^{6 x 6}
# Order: ['siah goshti', 'brown plus', 'brown momtaz', 'white mamooli', 'white momtaz', 'lux']
WALNUT_COST_MATRIX = np.array([
    [0.0, 2.5, 3.0, 4.0, 4.5, 5.0],  # siah goshti
    [2.5, 0.0, 1.0, 3.0, 3.5, 4.0],  # brown plus
    [3.0, 1.0, 0.0, 2.5, 3.0, 3.5],  # brown momtaz
    [4.0, 3.0, 2.5, 0.0, 1.0, 2.5],  # white mamooli
    [4.5, 3.5, 3.0, 1.0, 0.0, 2.0],  # white momtaz
    [5.0, 4.0, 3.5, 2.5, 2.0, 0.0]   # lux
], dtype=np.float32)


class Walnut(DatasetBase):
    dataset_dir = 'Walnut_Color_Parvizi_3'

    def __init__(self, root, num_shots):
        if root:
            if os.path.basename(os.path.normpath(root)) != self.dataset_dir:
                self.dataset_dir = os.path.join(root, self.dataset_dir)
            else:
                self.dataset_dir = root
        else:
            if os.path.exists(os.path.join('FewShotData', self.dataset_dir)):
                self.dataset_dir = os.path.join('FewShotData', self.dataset_dir)
            elif os.path.exists(os.path.join('..', 'FewShotData', self.dataset_dir)):
                self.dataset_dir = os.path.join('..', 'FewShotData', self.dataset_dir)
        self.template = template
        self._cost_matrix = torch.from_numpy(WALNUT_COST_MATRIX)
        
        train = self.read_data(os.path.join(self.dataset_dir, str(num_shots)))
        test = self.read_data(os.path.join(self.dataset_dir, 'test'))

        super().__init__(train_x=train, val=test, test=test)
    
    @property
    def cost_matrix(self):
        return self._cost_matrix

    def read_data(self, dir_path):
        items = []
        existing_dirs = sorted([d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))])
        # Preserve canonical order for classes present in the dataset
        classes = [c for c in CANONICAL_CLASSES if c in existing_dirs]
        for c in existing_dirs:
            if c not in classes:
                classes.append(c)

        label_map = {classname: i for i, classname in enumerate(classes)}
        
        for classname in classes:
            class_dir = os.path.join(dir_path, classname)
            label = label_map[classname]
            
            # Map Fingilish names to descriptive English for the prompt
            english_classname = CLASS_MAP.get(classname, classname)
            
            for img_name in sorted(os.listdir(class_dir)):
                if not img_name.startswith('.'):
                    impath = os.path.join(class_dir, img_name)
                    item = Datum(
                        impath=impath,
                        label=label,
                        classname=english_classname
                    )
                    items.append(item)
        return items

