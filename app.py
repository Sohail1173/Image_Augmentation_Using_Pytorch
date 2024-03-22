import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from pathlib import Path


class DataAugmentation:

    def __init__(self, data_dir:Path=r"C:\Users\91808\Downloads\data_car"):

        self.data_dir = data_dir
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224,224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Pad(10, fill=0),
            transforms.RandomRotation(10),
            transforms.Normalize(mean=[0.5, 0.6, 0.4], std=[0.5, 0.4, 0.7])
        ])

    def imagefolder(self):
        data = ImageFolder(self.data_dir, transform=self.transform)
        return data

    def data_loader(self, data_folder, batch_size=32, shuffle=True):
        data_loader = DataLoader(data_folder, batch_size=batch_size, shuffle=shuffle)
        return data_loader

    def show_images(self, data_loader):
        for images, labels in data_loader:
            for img in images:
                plt.imshow(img.permute(1, 2, 0))  # Convert from tensor shape to image shape
                plt.show()

    def initiate_transformation(self, batch_size=32, shuffle=True):
        image_dataset = self.imagefolder()
        data_loader = self.data_loader(image_dataset, batch_size=batch_size, shuffle=shuffle)
        self.show_images(data_loader)


obj = DataAugmentation()
obj.initiate_transformation()



    