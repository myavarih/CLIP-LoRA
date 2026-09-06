import os
from .utils import Datum, DatasetBase

template = ['a photo of a {} peanut, a type of peanut.']

CLASS_MAP = {
    'lape': 'split kernel',
    'salem': 'intact whole',
    'zayeat': 'defective waste'
}

class BadamZamini(DatasetBase):
    dataset_dir = 'BadamZamini_Chitgar_2'

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
            
            # Map Fingilish names to descriptive English for the prompt
            english_classname = CLASS_MAP.get(classname, classname)
            
            for img_name in os.listdir(class_dir):
                if not img_name.startswith('.'):
                    impath = os.path.join(class_dir, img_name)
                    item = Datum(
                        impath=impath,
                        label=label,
                        classname=english_classname
                    )
                    items.append(item)
        return items
