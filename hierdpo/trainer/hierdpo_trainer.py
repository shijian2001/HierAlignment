from openrlhf.trainer import DPOTrainer
from openrlhf.models import DPOLoss
from torch.optim import Optimizer


class HierDPOTrainer(DPOTrainer):
    """
    Hierarchical DPO Trainer with customizable loss function.

    Args:
        model (torch.nn.Module): The primary model to be trained.
        ref_model (torch.nn.Module): The reference model for comparing and guiding preference.
        strategy (Strategy): The strategy to use for training.
        tokenizer (Tokenizer): The tokenizer for processing input data.
        optim (Optimizer): The optimizer for training the model.
        train_dataloader (DataLoader): The dataloader for the training dataset.
        eval_dataloader (DataLoader): The dataloader for the evaluation dataset.
        scheduler (Scheduler): The learning rate scheduler to control learning rate during training.
        loss_cls (class, optional): The loss class to be used for computing the loss. Defaults to None.
        max_norm (float, defaults to 0.5): Maximum gradient norm for gradient clipping.
        beta (float, defaults to 0.01): Coefficient for regularizing the preference loss.
        max_epochs (int, defaults to 2): Maximum number of training epochs.
    """

    def __init__(
        self,
        model,
        ref_model,
        strategy,
        tokenizer,
        optim: Optimizer,
        train_dataloader,
        eval_dataloader,
        scheduler,
        loss_cls=None,
        max_norm=0.5,
        beta=0.01,
        max_epochs: int = 2,
    ) -> None:
        super().__init__(
            model,
            ref_model,
            strategy,
            tokenizer,
            optim,
            train_dataloader,
            eval_dataloader,
            scheduler,
            max_norm,
            beta,
            max_epochs,
        )
        if loss_cls is None:
            self.loss_fn = DPOLoss(self.beta, self.args.label_smoothing, self.args.ipo)
        else:
            self.loss_fn = loss_cls(self.beta, self.args.label_smoothing, self.args.ipo)
