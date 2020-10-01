# Auto-Augment

<p>
  ```python
  from simple-tool-pytorch import ImageNetPolicy, CIFAR10Policy, SVHNPolicy
  
  data = ImageFolder(rootdir, transform=transforms.Compose(
                        [transforms.RandomResizedCrop(224), 
                         transforms.RandomHorizontalFlip(), 
                         ImageNetPolicy(), 
                         transforms.ToTensor(),
                         transforms.Normalize(...)]))
  loader = DataLoader(data, ...)
  ```
</p>
