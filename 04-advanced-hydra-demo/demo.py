import hydra
import torch
import wandb

@hydra.main(config_path="configs", config_name="default", version_base=None)
def my_app(cfg):
    optim_partial = hydra.utils.instantiate(cfg.optimizer)
    scheduler_partial = hydra.utils.instantiate(cfg.scheduler)
    model:torch.nn.Module = torch.nn.Linear(1, 1)
    optim:torch.optim.Optimizer = optim_partial(model.parameters(), lr=cfg.train.lr)
    scheduler:torch.optim.lr_scheduler.LRScheduler = scheduler_partial(optim)

    # here we are using wandb in offline mode
    # to sync the results with wandb server later:
    # wandb sync --sync-all
    # wandb sync --include-offline wandb/offline-*
    wandb.init(
        mode='offline',
        project="different-lr-scheduling2", 
        config = {
            "lr": cfg.train.lr,
            "epochs": cfg.train.max_epochs,
        }
    )

    for i in range(cfg.train.max_epochs):
        optim.step()
        scheduler.step()
        lr = optim.param_groups[0]["lr"]
        wandb.log({"lr": lr, "epoch": i})

    wandb.finish()

if __name__ == "__main__":
    my_app()

# python demo.py -m train.lr=1e-3,2e-3,5e-3,1e-2,2e-2,5e-2,1e-1,2e-1,5e-1
    