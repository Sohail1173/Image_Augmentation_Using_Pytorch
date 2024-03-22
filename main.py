import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms




dir_path=r"C:\Users\91808\Downloads\data_car"


tranform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.Pad(10,fill=0),
    transforms.RandomRotation(10),
    transforms.Normalize(mean=[0.5,0.6,0.4],std=[0.5,0.4,0.7])
    
])


data=ImageFolder(dir_path,transform
                 =tranform)
data_loader=DataLoader(data,shuffle=True,batch_size=1)
print(type(data_loader))

for image,labels in data_loader:
    for img in image:
        # img=img.transpose(0,2)
        plt.imshow(img)
        plt.show()
        print(img.shape)
    print(image.shape)