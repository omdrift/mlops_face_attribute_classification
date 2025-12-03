import torch
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, images, labels, transform=None, is_preprocessed=False):
        """
        Args:
            images: numpy array
                - Si is_preprocessed=False: (N, H, W, C) - format numpy classique
                - Si is_preprocessed=True: (N, C, H, W) - déjà au format torch
            labels: numpy array (N, 5)
            transform: torchvision transforms
            is_preprocessed: True si les images viennent de train_data_s1.pt
        """
        if is_preprocessed:
            # Déjà au format (N, C, H, W), juste convertir en tensor
            self.images = torch.tensor(images, dtype=torch.float32)
        else:
            # Format (N, H, W, C) -> (N, C, H, W)
            self.images = torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2)
        
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)

        label_beard = self.labels[idx, 0].float()
        label_mustache = self.labels[idx, 1].float()
        label_glasses = self.labels[idx, 2].float()
        label_hair_color = self.labels[idx, 3]
        label_hair_length = self.labels[idx, 4]

        return image, {
            "beard": label_beard,
            "mustache": label_mustache,
            "glasses": label_glasses,
            "hair_color": label_hair_color,
            "hair_length": label_hair_length
        }