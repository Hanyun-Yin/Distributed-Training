#This file is to show how to use the accelerate library in Huggingface
import os, re, torch, PIL
import numpy as np

from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, RandomResizedCrop, Resize, ToTensor
from torchvision import models

from accelerate import Accelerator
from accelerate.utils import set_seed
from timm import create_model
from torch.optim.lr_scheduler import CosineAnnealingLR
from accelerate import notebook_launcher

def extract_label(fname):
    stem = fname.split(os.path.sep)[-1]
    return re.search(r"^(.*)_\d+\.jpg$", stem).groups()[0]


class PetsDataset(Dataset):
    def __init__(self, file_names, image_transform=None, label_to_id=None):
        self.file_names = file_names
        self.image_transform = image_transform
        self.label_to_id = label_to_id

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        fname = self.file_names[idx]
        raw_image = PIL.Image.open(fname)
        image = raw_image.convert("RGB")
        if self.image_transform is not None:
            image = self.image_transform(image)
        label = extract_label(fname)
        if self.label_to_id is not None:
            label = self.label_to_id[label]
        return {"image": image, "label": label}



def get_dataloaders(batch_size:int=64):
    "Builds a set of dataloaders with a batch_size"
    random_perm = np.random.permutation(len(fnames))
    cut = int(0.8 * len(fnames))
    train_split = random_perm[:cut]
    eval_split = random_perm[:cut]
    
    # For training we use a simple RandomResizedCrop
    train_tfm = Compose([
        RandomResizedCrop((224, 224), scale=(0.5, 1.0)),
        ToTensor()
    ])
    train_dataset = PetsDataset(
        [fnames[i] for i in train_split],
        image_transform=train_tfm,
        label_to_id=label_to_id
    )
    
    # For evaluation we use a deterministic Resize
    eval_tfm = Compose([
        Resize((224, 224)),
        ToTensor()
    ])
    eval_dataset = PetsDataset(
        [fnames[i] for i in eval_split],
        image_transform=eval_tfm,
        label_to_id=label_to_id
    )
    
    # Instantiate dataloaders
    train_dataloader = DataLoader(
        train_dataset, 
        shuffle=True, 
        batch_size=batch_size,
        num_workers=4
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        shuffle=False,
        batch_size=batch_size*2,
        num_workers=4
    )
    return train_dataloader, eval_dataloader



def training_loop(mixed_precision="fp16", seed:int=42, batch_size:int=64):
    set_seed(seed)
    # Initialize accelerator
    accelerator = Accelerator(mixed_precision=mixed_precision)
    # Build dataloaders
    train_dataloader, eval_dataloader = get_dataloaders(batch_size)
    
    # instantiate the model (we build the model here so that the seed also controls new weight initaliziations)
    #model = models.resnet50(pretrained=True)
    model = create_model("resnet50d", pretrained=True, num_classes=len(label_to_id))
    
    # Freeze the base model
    for param in model.parameters(): 
        param.requires_grad=False
    for param in model.get_classifier().parameters():
        param.requires_grad=True
        
    # We normalize the batches of images to be a bit faster
    mean = torch.tensor(model.default_cfg["mean"])[None, :, None, None]
    std = torch.tensor(model.default_cfg["std"])[None, :, None, None]
    
    # To make this constant available on the active device, we set it to the accelerator device
    mean = mean.to(accelerator.device)
    std = std.to(accelerator.device)
    
    # Intantiate the optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr = 3e-2/25)
    
    # Instantiate the learning rate scheduler
    lr_scheduler = OneCycleLR(
        optimizer=optimizer, 
        max_lr=3e-2, 
        epochs=5, 
        steps_per_epoch=len(train_dataloader)
    )
    
    # Prepare everything
    # There is no specific order to remember, we just need to unpack the objects in the same order we gave them to the
    # prepare method.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )
    
    # Now we train the model
    for epoch in range(5):
        model.train()
        for step, batch in enumerate(train_dataloader):
            # We could avoid this line since we set the accelerator with `device_placement=True`.
            batch = {k: v.to(accelerator.device) for k, v in batch.items()}
            inputs = (batch["image"] - mean) / std
            outputs = model(inputs)
            loss = torch.nn.functional.cross_entropy(outputs, batch["label"])
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        model.eval()
        accurate = 0
        num_elems = 0
        for _, batch in enumerate(eval_dataloader):
            # We could avoid this line since we set the accelerator with `device_placement=True`.
            batch = {k: v.to(accelerator.device) for k, v in batch.items()}
            inputs = (batch["image"] - mean) / std
            with torch.no_grad():
                outputs = model(inputs)
            predictions = outputs.argmax(dim=-1)
            accurate_preds = accelerator.gather(predictions) == accelerator.gather(batch["label"])
            num_elems += accurate_preds.shape[0]
            accurate += accurate_preds.long().sum()

        eval_metric = accurate.item() / num_elems
        # Use accelerator.print to print only on the main process.
        accelerator.print(f"epoch {epoch}: {100 * eval_metric:.2f}")

if __name__ == "__main__":
    data_dir = "./images"
    fnames = os.listdir(data_dir)
    # Grab all the image filenames
    fnames = [
        os.path.join(data_dir, fname)
        for fname in fnames
        if fname.endswith(".jpg")
    ]

    # Build the labels
    all_labels = [
        extract_label(fname)
        for fname in fnames
    ]
    id_to_label = list(set(all_labels))
    id_to_label.sort()
    label_to_id = {lbl: i for i, lbl in enumerate(id_to_label)}

    args = ("fp16", 42, 64)
    #notebook_launcher(training_loop, args, num_processes=4)
    training_loop()