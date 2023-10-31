import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import os
import json
from PIL import Image

class OralClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, annonations, transform=None):
        self.annonations = annonations
        self.transform = transform

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
        
        x, y, w, h = annotation["bbox"]
        subimage = image.crop((x, y, x+w, y+h))

        if self.transform:
            subimage = self.transform(subimage)

        category = self.categories[annotation["category_id"]]

        return subimage, category

    def get_image_id(self, idx):
        return self.dataset["images"][idx]["id"]


if __name__ == "__main__":
    import torchvision

    dataset = OralClassificationDataset(
        "data/oral1/train.json",
        transform=transforms.Compose([
            transforms.Resize((224, 224), antialias=True),
            transforms.ToTensor()
        ])
    )

    torchvision.utils.save_image(dataset[1][0], "test.png")