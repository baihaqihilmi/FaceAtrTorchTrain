import os
from PIL import Image
from torch.utils.data import Dataset

class AgeEstimationDataset(Dataset):
    def __init__(self, image_dir, labels_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        with open(labels_file, 'r') as file:
            for line in file:
                parts = line.strip().split()
                self.image_paths.append(parts[0])
                self.labels.append(int(parts[1]))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_paths[idx])
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label