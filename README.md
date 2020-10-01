# Auto-Augment

```python
from simple-tool-pytorch import ImageNetPolicy, CIFAR10Policy, SVHNPolicy
...

data = ImageFolder(rootdir, transform=transforms.Compose(
                      [transforms.RandomResizedCrop(224), 
                       transforms.RandomHorizontalFlip(), 
                       ImageNetPolicy(), # CIFAR10Policy(),  SVHNPolicy()
                       transforms.ToTensor(),
                       transforms.Normalize(...)]))
loader = DataLoader(data, ...)
...
```
source : https://github.com/DeepVoltaire/AutoAugment

# Warmup-Cosine-Lr
```python
from simple-tool-pytorch import GradualWarmupScheduler
...

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)
cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0, last_epoch=-1)
scheduler = GradualWarmupScheduler(optim, multiplier=10, total_epoch=5, after_scheduler=cosine_scheduler)

for i, (images, labels) in enumerate(train_data):
    ...

    scheduler.step(epoch) # Last position
```
source : https://github.com/seominseok0429/pytorch-warmup-cosine-lr

# Mixup

```python
from simple-tool-pytorch import mixup_data, mixup_criterion
...

alpha = 0.2  # set beta distributed parm, 0.2 is recommend.
criterion = torch.nn.CrossEntropyLoss()

for i, (images, labels) in enumerate(train_data):
    images = images.cuda()
    labels = labels.cuda()

    data, labels_a, labels_b, lam = mixup_data(images, labels, alpha)
    optimizer.zero_grad()
    outputs = model(images)
    loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)

    loss.backward()
    optimizer.update()
    ...
```

# Label-Smoothing

```python
from simple-tool-pytorch import LabelSmoothingCrossEntropy
...

criterion = LabelSmoothingCrossEntropy()
...

for i, (images, labels) in enumerate(train_data):
    ...

    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    ...
```

source : https://github.com/seominseok0429/label-smoothing-visualization-pytorch
