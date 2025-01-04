# hier_alignment

## Environment
You can enter into the ```openrlhf``` conda env as follows:
```
conda activate openrlhf
```

## Launching DPO training
You can start a DPO training as follows:
```
bash train_dpo_llama.sh
```
You should specify the ```--loss```, otherwise it will use the default ```DPOLoss```.

## Add a custom loss
First you should inherit ```BaseLoss``` to implement your custom Loss Class as follows:
```python
class BaseLoss(nn.Module, ABC):
    """
    Abstract Loss Base Class
    """

    def __init__(self, beta: float, label_smoothing: float = 0.0, ipo: bool = False) -> None:
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.ipo = ipo

    @abstractmethod
    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pass


# Default DPO Loss
class DefaultDPOLoss(BaseLoss):
    """
    Default DPOLoss Implementation
    """

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        logits = pi_logratios - ref_logratios

        if self.ipo:
            losses = (logits - 1 / (2 * self.beta)) ** 2
        else:
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )

        loss = losses.mean()
        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()

        return loss, chosen_rewards, rejected_rewards
```
Then you should register your implemented loss as follows:
```python
# Registering all loss classes
LOSSES = {
    "default": "DefaultDPOLoss",
    ## Please add a custome loss
}
```
In the ```train_dpo_llama.sh```, you can specify the ```--loss``` via the key in the ```LOSSES``` dict.