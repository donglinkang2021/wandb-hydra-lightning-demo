import hydra
from config import Config
from tqdm import tqdm
import time

@hydra.main(config_path="conf", config_name="default", version_base=None)
def my_app(cfg: Config) -> None:
    # print(OmegaConf.to_yaml(cfg))
    print(f"model: {cfg.model.name} dataset: {cfg.dataset.name} optimizer: {cfg.optimizer.name}")

    # just simulate the training loop
    pbar = tqdm(total=cfg.training.epochs)
    for epoch in range(cfg.training.epochs):
        pbar.update(1)
    time.sleep(0.1)
    pbar.close()

if __name__ == "__main__":
    my_app()