from cmath import nan
from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch import Tensor, nn


def compute_l1_loss(targets: Tensor, predictions: Tensor) -> Tensor:
    """Compute the mean absolute error (MAE)/L1 loss between `predictions` and `targets`.

    Specifically, the l1-weighted MSE loss can be computed as follows:
    1. Compute a binary mask of the `targets` that are not NaN, and apply it to the `targets` and `predictions`
    2. Compute the MAE loss between `predictions` and `targets`.
        This should give us a [batch_size * num_actors x T x 2] tensor `l1_loss`.
    3. Compute the mean of `l1_loss`. This gives us our final scalar loss.

    Args:
        targets: A [batch_size * num_actors x T x 2] tensor, containing the ground truth targets.
        predictions: A [batch_size * num_actors x T x 2] tensor, containing the predictions.

    Returns:
        A scalar MAE loss between `predictions` and `targets`
    """
    # TODO: Implement.
    #nan_mask = ~targets.isnan()
    #l1_loss = torch.abs((targets * nan_mask) - (predictions * nan_mask))
    return torch.mean(l1_loss)
    l1_loss = torch.abs(targets - predictions)
    return l1_loss[~l1_loss.isnan()].mean()


def compute_nll_loss(targets: Tensor, predictions: Tensor) -> Tensor:
    """compute the negative log likelihoood loss between 'predictions' and 'targets'

    Args:
        targets: A [batch_size * num_actors x T x 2] tensor, containing the ground truth targets.
        predictions: A [batch_size * num_actors x T x 5] tensor, containing the predictions. (mu_x, mu_y, rho, sigma_x, sigma_y)

    Returns:
        A scalar nll loss between `predictions` and `targets`
    """
    BN, T, _ = targets.shape

    targets = torch.where(targets.isnan(), torch.tensor(0, dtype=targets.dtype), targets)

    rhos = torch.clamp(predictions[:,:,2:3], min=-1.0, max=1.0)

    cov1 = torch.stack((predictions[:,:,3:4] ** 2, rhos * predictions[:,:,3:4] * predictions[:,:,4:5]), dim=3)
    cov2 = torch.stack((rhos * predictions[:,:,3:4] * predictions[:,:,4:5], predictions[:,:,4:5] ** 2), dim=3)
    cov = torch.cat((cov1, cov2), dim=2)

    tmm = torch.unsqueeze(targets - predictions[..., 0:2], dim=3)

    loss = torch.log(torch.norm(cov, dim=(2,3))) + (tmm.transpose(dim0=2,dim1=3) @ cov.inverse() @ tmm).squeeze()
    loss = 0.5 * loss
    loss = torch.sum(loss, dim=1)
    return loss.nanmean()


@dataclass
class PredictionLossConfig:
    """Prediction loss function configuration.

    Attributes:
        l1_loss_weight: The multiplicative weight of the L1 loss
    """

    l1_loss_weight: float


@dataclass
class PredictionLossMetadata:
    """Detailed breakdown of the Prediction loss."""

    total_loss: torch.Tensor
    l1_loss: torch.Tensor


class PredictionLossFunction(torch.nn.Module):
    """A loss function to train a Prediction model."""

    def __init__(self, config: PredictionLossConfig) -> None:
        super(PredictionLossFunction, self).__init__()
        self._l1_loss_weight = config.l1_loss_weight

    def forward(
        self, predictions: List[Tensor], targets: List[Tensor]
    ) -> Tuple[torch.Tensor, PredictionLossMetadata]:
        """Compute the loss between the predicted Predictions and target labels.

        Args:
            predictions: A list of batch_size x [num_actors x T x 2] tensor containing the outputs of
                `PredictionModel`.
            targets:  A list of batch_size x [num_actors x T x 2] tensor containing the ground truth output.

        Returns:
            The scalar tensor containing the weighted loss between `predictions` and `targets`.
        """
        predictions_tensor = torch.cat(predictions)
        targets_tensor = torch.cat(targets)

        # 1. Unpack the targets tensor.
        target_centroids = targets_tensor[..., :2]  # [batch_size * num_actors x T x 2]

        # 2. Unpack the predictions tensor.
        predicted_centroids = predictions_tensor[
            ..., :5
        ]  # [batch_size * num_actors x T x 5]

        # 3. Compute individual loss terms for l1
        #l1_loss = compute_l1_loss(target_centroids, predicted_centroids)
        l1_loss = compute_nll_loss(target_centroids, predicted_centroids)

        # 4. Aggregate losses using the configured weights.
        total_loss = l1_loss * self._l1_loss_weight

        loss_metadata = PredictionLossMetadata(total_loss, l1_loss)
        return total_loss, loss_metadata
