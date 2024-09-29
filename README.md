<div align=center>

# WandB+Hydra+Lightning Examples

[![wandb](https://img.shields.io/badge/-Weights%20&%20Biases-FFBE00?style=flat-square&logo=weightsandbiases&logoColor=FFFFFF)](https://wandb.ai/)
[![hydra](https://img.shields.io/badge/-Hydra-89B8CD?style=flat-square&logoColor=FFFFFF)](https://hydra.cc/)
[![lightning](https://img.shields.io/badge/-Lightning-792EE5?style=flat-square&logo=lightning&logoColor=FFFFFF)](https://lightning.ai/)

</div>

This is a simple example of how to use [WandB](https://wandb.ai/), [Hydra](https://hydra.cc/) and [PyTorch Lightning](https://lightning.ai/) together.

## Installation

```shell
pip install hydra-core --upgrade
pip install wandb
```

## 00-wandb-Demo

First, you need to login to WandB, and open the link to get your API key.

```shell
(GPT) root@container:~/wandb-hydra-demo# wandb login
wandb: Logging into wandb.ai. (Learn how to deploy a W&B server locally: https://wandb.me/wandb-server)
wandb: You can find your API key in your browser here: https://wandb.ai/authorize
wandb: Paste an API key from your profile and hit enter, or press ctrl+c to quit: 
wandb: Appending key for api.wandb.ai to your netrc file: /root/.netrc
```

Then you can run the example code.

```shell
cd 00-wandb-demo
python demo.py
```

You will see the results in the WandB dashboard.

## 01-hydra-demo

> refer to https://hydra.cc/docs/intro/

In this example, we will use Hydra to manage the configuration of

- different datasets
- different models
- different optimizers
- and you can add more settings here, such as different hardware, logging, training, etc.

And you can run a bunch of experiments with different configurations just use a simple command.

```shell
cd 01-hydra-demo
python main.py --multirun \
    dataset=mnist,cifar10,cifar100 \
    model=resnet18,resnet34,resnet50,resnet101,resnet152 \
    optimizer=adam,sgd
```

You will see the outputs in the console like this:

```shell
[2024-09-26 17:00:29,062][HYDRA]        #0 : dataset=mnist model=resnet18 optimizer=adam
model: resnet18 dataset: mnist optimizer: adam
100%|████████████████████████| 200/200 [00:00<00:00, 1991.40it/s]
...
[2024-09-26 17:10:08,740][HYDRA]        #29 : dataset=cifar100 model=resnet152 optimizer=sgd
model: resnet152 dataset: cifar100 optimizer: sgd
100%|████████████████████████| 200/200 [00:00<00:00, 1992.22it/s]
```

## 02-lightning-demo

In this example, we just use `lightning`, `torch`, `torchvision` and `torchmetrics` to train a simple model.

- `train.py` is the main script to train the model.
- `model.py` contains the `LightningModule` class `ImagenetTransferLearning`, just used to train a pretrained `resnet50` on the CIFAR-10 dataset.
- `dataset.py` contains the `DataModule` class `CIFAR10DataModule`, used to load the CIFAR-10 dataset.
- `config.py` contains the configuration of the training process.

```shell
python 02-lightning-demo/train.py
```

## 03-wandb-hydra-lightning-demo

In this example, we combine the above three examples together.

- `main.py` is the main script to train the model.
- `configs` contains the `.yaml` configuration.
- `src` contains the source `lightning` code to train the model.

> You need to login to WandB before running the code.

```shell
wandb login
```

Run different backbones to train the image classification model on CIFAR-10 dataset.

```shell
cd 03-wandb-hydra-lightning-demo
python main.py --multirun \
    module.backbone=resnet18,resnet34,resnet50
```

You will see the results in the WandB dashboard.

| train accuracy | val accuracy |
| :-------------: | :----------: |
| ![alt](images/README/W&B%20Chart%202024_9_29%2018_12_44.png) | ![alt](images/README/W&B%20Chart%202024_9_29%2018_14_29.png) |

# 04-advanced-hydra-demo

In this example, we use the `hydra.utils.instantiate` function to instantiate the class, so you can write the configuration in a more concise way, like this:

```yaml
optimizer:
    _partial_: true
    _target_: torch.optim.SGD
    momentum: 0.9
    weight_decay: 1e-4
```

- `_partial_: true` means that the configuration is a partial configuration, and the `optim_partial` below is just a **function** that needs to be called with the parameters.
- `_target_: torch.optim.SGD` means that the `optimizer` is a `torch.optim.SGD` class.

```python
optim_partial = hydra.utils.instantiate(cfg.optimizer)
optim:torch.optim.Optimizer = optim_partial(model.parameters(), lr=cfg.train.lr)
```

And you can run the following command to train the model with different learning rates.

```shell
python demo.py -m train.lr=1e-3,2e-3,5e-3,1e-2,2e-2,5e-2,1e-1,2e-1,5e-1
```

## Notes

If you find that run the `wandb` script in the online mode (default) is too slow, you can use the offline mode by setting the `mode` parameter to `offline`.

```python
wandb.init(
    mode='offline',
    ...
)
```

And you can sync the offline data to the online server by running the following command.

```shell
# just select one of the following commands, refer to https://github.com/wandb/wandb/issues/3111
wandb sync --sync-all
wandb sync --include-offline wandb/offline-*
```