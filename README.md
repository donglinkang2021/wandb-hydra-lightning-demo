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