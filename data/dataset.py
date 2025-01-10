import os
from PIL import Image
from torch.utils.data import Dataset
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

class UTKDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.file_list = [f for f in os.listdir(root_dir) if f.endswith('.jpg')]
        self.labels = {
            "White": 0,
            "Black": 1,
            "Asian": 2,
            "Indian": 3,
            "Others": 4
        }


    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = os.path.join(self.root_dir, self.file_list[idx])
        image = Image.open(img_name).convert('RGB')

        # Parse filename to get labels
        file_name = os.path.basename(img_name)
        try:
            age, gender, race, *_ = file_name.split('_')
            age = torch.tensor(int(age), dtype=torch.float32)
            gender = torch.tensor(int(gender), dtype=torch.float32)
            race = torch.tensor(int(race), dtype=torch.float32)
        except ValueError as e:
            raise ValueError(f"Filename {file_name} does not match the expected pattern.") from e

        label = {"Age": age, "Gender": gender, "Race": race}

        if self.transform:
            image = self.transform(image)

        return image, label

if __name__ == "__main__":
    import torchvision.transforms as transforms

    # Define a simple transform to convert images to tensor
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    # Create the dataset
    dataset = UTKDataset(root_dir=r'data\archive\UTKFace', transform=transform)

    # Create a DataLoader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

    # Iterate through the dataset
    for i, (images, labels) in enumerate(dataloader):
        print(f"Batch {i+1}")
        print(f"Images: {images.shape}")
        print(f"Labels: {labels}")
        if i == 1:  # Just to limit the output for demonstration
            break