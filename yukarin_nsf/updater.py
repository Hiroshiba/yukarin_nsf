from typing import Optional

from pytorch_trainer.dataset import convert
from pytorch_trainer.training import StandardUpdater
from torch.optim.optimizer import Optimizer

from yukarin_nsf.model import DiscriminatorModel, Model


class Updater(StandardUpdater):
    def __init__(
        self,
        iterator,
        optimizer: Optimizer,
        discriminator_optimizer: Optional[Optimizer],
        model: Model,
        discriminator_model: Optional[DiscriminatorModel],
        device,
    ):
        optimizers = dict(main=optimizer)
        if discriminator_optimizer is not None:
            optimizers["discriminator"] = discriminator_optimizer

        models = dict(main=model)
        if discriminator_model is not None:
            models["discriminator"] = discriminator_model

        super().__init__(
            iterator=iterator, optimizer=optimizers, model=models, device=device,
        )

    def contain_discriminator(self):
        return "discriminator" in self._optimizers

    def update_core(self):
        if not self.contain_discriminator():
            return super().update_core()

        iterator = self._iterators["main"]
        batch = iterator.next()
        in_arrays = convert._call_converter(self.converter, batch, self.device)

        optimizer = self._optimizers["main"]
        model = self._models["main"]
        discriminator_optimizer = self._optimizers["discriminator"]
        discriminator_model = self._models["discriminator"]

        for m in self._models.values():
            m.train()

        # discriminator
        discriminator_optimizer.zero_grad()
        discriminator_model(**in_arrays).backward()
        discriminator_optimizer.step()

        # predictor
        optimizer.zero_grad()
        model(**in_arrays).backward()
        optimizer.step()
