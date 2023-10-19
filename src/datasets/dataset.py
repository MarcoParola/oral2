import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import os
import json
from matplotlib import pyplot as plt
from PIL import Image

class OralClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, annonations, crop, transform=None):
        self.annonations = annonations
        self.transform = transform
        self.crop = crop

        with open(annonations, "r") as f:
            self.dataset = json.load(f)
        
        self.images = dict()
        for image in self.dataset["images"]:
            self.images[image["id"]] = image
        
        self.categories = dict()
        for i, category in enumerate(self.dataset["categories"]):
            self.categories[category["id"]] = i

        
    def __len__(self):
        return len(self.dataset["annotations"])

    def __getitem__(self, idx):
        annotation = self.dataset["annotations"][idx]
        image = self.images[annotation["image_id"]]
        image_path = os.path.join(os.path.dirname(self.annonations), "oral1", image["file_name"])
        image = Image.open(image_path).convert("RGB")

        # if crop set to True image is cropped
        if self.crop:
            x, y, w, h = annotation["bbox"]
            subimage = image.crop((x, y, x+w, y+h))
        # if crop set to False image is not cropped
        else:
            subimage = image

        if self.transform:
            subimage = self.transform(subimage)

        category = self.categories[annotation["category_id"]]

        return subimage, category


if __name__ == "__main__":
    import torchvision

    dataset_cropped = OralClassificationDataset(
        "data/train.json", True,
        transform=transforms.Compose([
            transforms.Resize((224, 224), antialias=True),
            transforms.ToTensor()
        ])
    )
    image_cropped, mask = dataset_cropped.__getitem__(5)
    plt.imshow(image_cropped.permute(1, 2, 0))
    plt.savefig("cropped_test.png")


    dataset_not_cropped = OralClassificationDataset(
        "data/train.json", False,
        transform=transforms.Compose([
            transforms.Resize((224, 224), antialias=True),
            transforms.ToTensor()
        ])
    )
    image_not_cropped, mask = dataset_not_cropped.__getitem__(5)
    plt.imshow(image_not_cropped.permute(1, 2, 0))
    plt.savefig("not_cropped_test.png")
    #torchvision.utils.save_image(dataset[1][0], "test.png")