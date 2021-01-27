# Simple_Tool_Pytorch

## Getting Started
```Shell
cd (yourprojectdir)
git clone https://github.com/cjf8899/simple_tool_pytorch.git

```
## Summary
- [Auto-Augment](https://github.com/cjf8899/simple_tool_pytorch#Auto-Augment)
- [Warmup-Cosine-Lr](https://github.com/cjf8899/simple_tool_pytorch#Warmup-Cosine-Lr)
- [Mixup](https://github.com/cjf8899/simple_tool_pytorch#Mixup)
- [Label-Smoothing](https://github.com/cjf8899/simple_tool_pytorch#Label-Smoothing)
- [Random-erasing-augmentation](https://github.com/cjf8899/simple_tool_pytorch#Random-erasing-augmentation)
- [Focal-Loss](https://github.com/cjf8899/simple_tool_pytorch#Focal-Loss)


## Auto-Augment

```python
from simple_tool_pytorch import ImageNetPolicy, CIFAR10Policy, SVHNPolicy
...

data = ImageFolder(rootdir, transform=transforms.Compose(
                      [transforms.RandomResizedCrop(256), 
                       transforms.RandomHorizontalFlip(), 
                       ImageNetPolicy(), # CIFAR10Policy(),  SVHNPolicy()
                       transforms.ToTensor(),
                       transforms.Normalize(...)]))
loader = DataLoader(data, ...)
...
```
source : https://github.com/DeepVoltaire/AutoAugment

## Warmup-Cosine-Lr
```python
from simple_tool_pytorch import GradualWarmupScheduler
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

## Mixup

```python
from simple_tool_pytorch import mixup_data, mixup_criterion
...

alpha = 0.2  # set beta distributed parm, 0.2 is recommend.
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)

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

## Label-Smoothing

```python
from simple_tool_pytorch import LabelSmoothingCrossEntropy
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

## Random-erasing-augmentation

```python
from simple_tool_pytorch import RandomErasing
...

erasing_percent = 0.5
data = ImageFolder(rootdir, transform=transforms.Compose(
                      [transforms.RandomHorizontalFlip(), 
                       transforms.ToTensor(),
                       transforms.Normalize((0.4914, 0.4822, 0.4465), (...)),
                       RandomErasing(probability=erasing_percent, mean=[0.4914, 0.4822, 0.4465])]))
loader = DataLoader(data, ...)

```

## Focal-Loss
ex) Multiboxloss for SSD

```python
from simple_tool_pytorch import FocalLoss
...

    ...
    pos_idx = pos.unsqueeze(2).expand_as(conf_data)
    neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        
        
    conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
    targets_weighted = conf_t[(pos+neg).gt(0)]
        
    ###Focal loss
    compute_c_loss = FocalLoss(alpha=None, gamma=2, class_num=num_classes, size_average=False)
    loss_c = compute_c_loss(conf_p, targets_weighted)

    # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N

    N = num_pos.data.sum()
    loss_l /= N
    loss_c /= N
    return loss_l, loss_c


```
