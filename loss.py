from typing import Tuple, Dict, Type
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod

# Registering all loss classes
LOSSES = {
    "default": "DefaultDPOLoss",
    ## Please add a custome loss
}

def list_losses():
    return list(LOSSES.keys())

# Abstraction
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
        """
        Forward method to compute the loss.

        Args:
            policy_chosen_logps: Log probabilities of the chosen policy actions.
            policy_rejected_logps: Log probabilities of the rejected policy actions.
            reference_chosen_logps: Log probabilities of the chosen reference actions.
            reference_rejected_logps: Log probabilities of the rejected reference actions.

        Returns:
            A tuple of (loss, chosen_rewards, rejected_rewards).
        """
        pass

######################################################### Implemented Loss #########################################################

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
            losses = (logits - 1 / (2 * self.beta)) ** 2  # Eq. 17 of https://arxiv.org/pdf/2310.12036v2.pdf
        else:
            # Eq. 3 https://ericmitchell.ai/cdpo.pdf; label_smoothing=0 gives original DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
            losses = (
                # pylint: disable=not-callable
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )

        loss = losses.mean()
        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()

        return loss, chosen_rewards, rejected_rewards