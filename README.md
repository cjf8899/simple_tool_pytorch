# Auto-Augment


```python
from simple-tool-pytorch import ImageNetPolicy, CIFAR10Policy, SVHNPolicy

data = ImageFolder(rootdir, transform=transforms.Compose(
                      [transforms.RandomResizedCrop(224), 
                       transforms.RandomHorizontalFlip(), 
                       ImageNetPolicy(), # CIFAR10Policy(),  SVHNPolicy()
                       transforms.ToTensor(),
                       transforms.Normalize(...)]))
loader = DataLoader(data, ...)
```
source : <a link =https://github.com/DeepVoltaire/AutoAugment>https://github.com/DeepVoltaire/AutoAugment</a>

https://github.com/DeepVoltaire/AutoAugment
