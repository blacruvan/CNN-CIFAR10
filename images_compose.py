from torchvision import transforms

class TransformConfig:
    def __init__(self, image_size=32, mean=None, std=None):
        if mean is None:
            mean = [0.4914, 0.4822, 0.4465]
        if std is None:
            std = [0.247, 0.243, 0.261]
        
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.composed_train = self._create_composed_train_transform()
        self.composed_test = self._create_composed_test_transform()

    def _create_composed_train_transform(self):
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(0.1),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1), 
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.1),
            transforms.ToTensor(), 
            transforms.Normalize(self.mean, self.std),
            transforms.RandomErasing(p=0.75, scale=(0.02, 0.1), value=1.0, inplace=False)
        ])

    def _create_composed_test_transform(self):
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])