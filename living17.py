from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

class Living17(Dataset):
    def __init__(self, root="/usr/workspace/trivedi1/vision_data/breeds/", 
            train_path_file="/usr/workspace/trivedi1/vision_data/Living-17/living17_source_val.csv",
            test_path_file="/usr/workspace/trivedi1/vision_data/Living-17/living17_target_val.csv",
            transform=None,
            split="train"):
        super().__init__()

        train_df = pd.read_csv(train_path_file)
        train_df.path = train_df.path.str.replace("/p/lscratchh/jjayaram/BREEDS/breeds_data/",root,regex=True)
        self.train_files = list(train_df.itertuples(index=False, name=None))[1:] #dropping the comment! 
        
        test_df = pd.read_csv(test_path_file)
        test_df.path = test_df.path.str.replace("/p/lscratchh/jjayaram/BREEDS/breeds_data/",root,regex=True)

        self.test_files = list(test_df.itertuples(index=False, name=None))[1:] #dropping the comment! 
        self._transform = transform

        self.means = [0.485, 0.456, 0.406]
        self.stds = [0.228, 0.224, 0.225]

        self.split = split 
        if self.split == 'train':
            self.data = self.train_files
        elif self.split == 'test':
            self.data = self.test_files
        else:
            print("ERROR- INVALID SPLIT; EXITING!")
            exit()
    def __getitem__(self, i):

        path, y = self.data[i]
        x = Image.open(path)
        x = x.convert('RGB')
        if self._transform is not None:
            x = self._transform(x)
        return x, y

    def __len__(self) -> int:
        return len(self.data)

    def get_num_classes(self):
        return 17 