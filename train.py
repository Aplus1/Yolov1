import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as FT
import torch.optim as optim
import tqdm
from torch.utils.data import DataLoader
from model import Yolov1
from dataset import VOCDataset

from loss import YoloLoss

seed = 123
torch.manual_seed(seed)

#hyper params
Lr = 2e-5
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 16
weight_decay = 0
epochs = 100
num_workers = 2
pin_memory = True
load_model = False

#file paths
load_model_file = "model.pth"
image_dir = "data/images"
label_dir = "data/labels"

class Compose(object):

    def __init__(self,transform):
        self.transforms = transform

    def __call__(self,img,bboxes):
        for trans in self.transforms:
            #Note : if you wanted to rotate the image
            # then you have to apply the transform to the bboxes as well
            # so that it alse rotates the coordinates as well
            img,bboxes = trans(img),bboxes

        return img,bboxes

transform = Compose([transforms.Resize((448,448)),transforms.ToTensor()])

def train_fn(train_loader,model,optimizer,loss_fn):
    loop = tqdm(train_loader,leave=True)
    mean_loss = []

    for batch_idx,(x , y) in enumerate(loop):
        x,y = x.to(device), y.to(device)
        out = model(x)

        loss = loss_fn(out,y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_posix(loss=loss.item())

    print(f"Mean loss was {sum(mean_loss)/len(mean_loss)}")


def main():
    model = Yolov1(split_size = 7, num_boxes = 2,num_classes = 20).to(device)
    optimizer = optim.Adam(model.parameters(),lr=Lr,weight_decay=weight_decay)
    loss_fn = YoloLoss()

    if load_model:
        load_checkpoint(torch.load(load_model_file),model,optimizer)

    train_dataset = VOCDataset("data/100examples.csv",
                              transform=transform,
                              image_dir=image_dir,
                              label_dir=label_dir
                              )

    test_dataset = VOCDataset("data/test.csv",
                              transform=transform,
                              image_dir=image_dir,
                              label_dir=label_dir
                              )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
        drop_last=False
    )

    test_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
        drop_last=True
    )

    for epoch in range(epochs):
        pred_boxes,target_boxes = get_bboxes(
            train_loader,model,iou_threshold=0.5,threshold=0.4
        )

        mean_avg_prec = mean_avg_precision(
            pred_boxes,target_boxes,iou_threshold=0.5,box_format="midpoint"
        )

        print(f"Train mAP : {mean_avg_prec}")

        train_fn(train_loader,model,optimizer,loss_fn)

if __name__ == "__main__":
    main()