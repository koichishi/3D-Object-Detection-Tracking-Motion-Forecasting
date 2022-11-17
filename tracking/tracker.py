from copy import deepcopy
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import Tensor

from tracking.cost import iou_2d, Ciou_2d
from tracking.matching import greedy_matching, hungarian_matching
from tracking.types import ActorID, AssociateMethod, SingleTracklet


class Tracker:
    """Basic online tracker that tracks consecutive frames"""

    def __init__(
        self,
        track_steps: int,
        associate_method: AssociateMethod,
        min_score: float = 0.3,  # we only match bbox with confidence score >= min_score
        match_th: float = 1.0,  # we only filter out matches with cost >= match_th
        device: str = "cuda",
    ):
        assert (
            track_steps >= 2
        ), f"We should track at least 2 frames, got track_steps={track_steps}"
        self.track_steps = track_steps
        self.associate_method = associate_method
        self.min_score = min_score
        self.match_th = match_th
        self.reset()
        self.device = device

    def reset(self):
        """Reset the Tracker"""
        self.num_tracks = 0
        self.next_track_id = 0
        self.tracks: Dict[ActorID, SingleTracklet] = {}

    def create_new_tracklet(self, frame_id, bbox, score) -> int:
        """Given the initial frame_id, bbox and score, create a new tracklet and add to self.tracks"""
        self.num_tracks += 1
        new_tracklet = SingleTracklet([frame_id], [bbox], [score])
        new_track_id = self.next_track_id
        self.tracks[new_track_id] = new_tracklet
        self.next_track_id += 1
        return new_track_id

    def cost_matrix(self, bboxes1: Tensor, bboxes2: Tensor) -> Tensor:
        """Given two set of bounding boxes, this function computes the affinity matrix between two bbox sets

        Args:
            bboxes1: bounding box set of shape [M, 5]
            bboxes2: bounding box set of shape [N, 5]
        Returns:
            cost_matrix: cost matrix of shape [M, N]
        """
        # M, N = bboxes1.shape[0], bboxes2.shape[0]
        # cost_matrix = torch.ones((M, N))

        # improvement: uncomment when applying CIoU
        # iou_penalty = Ciou_2d(bboxes1, bboxes2)
        # return 1 - iou_penalty
        
        # improvement: uncomment when applying velocity predition or original implementation
        return 1 - iou_2d(bboxes1, bboxes2)

    def velocity_prediction(self, prev_bbox_ids):
        velocities = []
        for track_id in prev_bbox_ids:
            tracklet = self.tracks[track_id].bboxes_traj
            cur_bbox = tracklet[-1][: 2] # Only keep centroid
            if len(tracklet) > 1:
                prev_bbox = tracklet[-2][: 2] # Only keep centroid
            else:
                prev_bbox = 0
            velocities.append(cur_bbox - prev_bbox)
        # Padding back to Mx5
        pad = torch.nn.ZeroPad2d((0, 3, 0, 0))
        velocities = pad(torch.stack(velocities))
        # print(velocities)
        return velocities

    def bbox_prediction(self, velocities: Tensor, position: Tensor):
        return position + velocities

    def associate_greedy(
        self, bboxes1: Tensor, bboxes2: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """This function aims to find the greedy association between two set of bounding boxes

        Args:
            bboxes1: bounding box set of shape [M, 5]
            bboxes2: bounding box set of shape [N, 5]
        Returns:
            assign_matrix: binary assignment matrix of shape [M, N], where A[i,j] = 1 means i-th box in bboxes1
            and j-th box in bboxes2 are associated.
            cost_matrix: cost matrix of shape [M, N]
        """
        M, N = bboxes1.shape[0], bboxes2.shape[0]
        cost_matrix = self.cost_matrix(bboxes1, bboxes2)
        assign_matrix = torch.zeros((M, N))
        row_ids, col_ids = greedy_matching(cost_matrix)
        assign_matrix[row_ids, col_ids] = 1
        return assign_matrix, cost_matrix

    def associate_hungarian(
        self, bboxes1: Tensor, bboxes2: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """This function aims to find the hungarian association between two set of bounding boxes

        Args:
            bboxes1: bounding box set of shape [M, 5]
            bboxes2: bounding box set of shape [N, 5]
        Returns:
            assign_matrix: binary assignment matrix of shape [M, N], where A[i,j] = 1 means i-th box in bboxes1
            and j-th box in bboxes2 are associated.
            cost_matrix: cost matrix of shape [M, N]
        """
        M, N = bboxes1.shape[0], bboxes2.shape[0]
        cost_matrix = self.cost_matrix(bboxes1, bboxes2)
        assign_matrix = torch.zeros((M, N))
        row_ids, col_ids = hungarian_matching(cost_matrix)
        assign_matrix[row_ids, col_ids] = 1
        return assign_matrix, cost_matrix

    def track_consecutive_frame(
        self, bboxes1: Tensor, bboxes2: Tensor, predicted_bboxes2=None
    ) -> Tuple[Tensor, Tensor]:
        """This function tracks the bboxes2 in the current frame against bboxes1 in the previous frame,
        by associating bboxes1 and bboxes2 with associate_method, and filtering out the associations with
        matching score lower than self.match_th.

        Args:
            bboxes1: bounding box set of shape [M, 5]
            bboxes2: bounding box set of shape [N, 5]
            predicted_bboxes2: bounding box set of shape [M, 5]
        Returns:
            assign_matrix: binary assignment matrix of shape [M, N], where A[i,j] = 1 means i-th box in bboxes1
            and j-th box in bboxes2 are associated. For any 0 <= i < M, sum_{0 <= j < n}(A[i,j]) is either 0
            or 1. If A[i, j] = 0 for all j, then bboxes1[i] is the end of a tracklet. If A[i, j] = 0 for all i,
            then bboxes2[j] is the start of a new tracklet.
            cost_matrix: cost matrix of shape [M, N]
        """
        if predicted_bboxes2 is not None:
            bboxes1 = predicted_bboxes2

        if self.associate_method == AssociateMethod.GREEDY:
            assign_matrix, cost_matrix = self.associate_greedy(bboxes1, bboxes2)
        elif self.associate_method == AssociateMethod.HUNGARIAN:
            assign_matrix, cost_matrix = self.associate_hungarian(bboxes1, bboxes2)
        else:
            raise ValueError(f"Unknown association method {self.associate_method}")

        # Uncomment if check association similarity rates between the two algorithm
        # assign_matrix, _ = self.associate_greedy(bboxes1, bboxes2)
        # assign_matrix1, cost_matrix = self.associate_hungarian(bboxes1, bboxes2)

        assign_matrix = torch.where(torch.tensor(cost_matrix >= self.match_th), 
                    torch.tensor(0.0), assign_matrix)
        # Uncomment if check association similarity rates between the two algorithm
        # assign_matrix1 = torch.where(torch.tensor(cost_matrix) >= self.match_th, 
        #             torch.tensor(0.0), assign_matrix1)
        # print("assignments the same? ", torch.equal(assign_matrix1, assign_matrix))
        return assign_matrix, cost_matrix

    def track(self, bboxes_seq: List[Tensor], scores_seq: List[Tensor]):
        """Perform tracking given a sequence of bboxes and bbox confidence scores.
        We only track bboxes with scores >= self.min_score.

        Args:
            bboxes_seq: sequence of bounding box set of shape [N_i, 5]
            scores_seq: sequences of bounding box confidence scores of shape [N_i, ]
        Returns:
            None
        """
        total_assign_diff = []
        cur_frame_track_ids = []
        for idx, bbox in enumerate(bboxes_seq[0]):
            if scores_seq is not None and scores_seq[0][idx] < self.min_score:
                continue
            new_track_id = self.create_new_tracklet(0, bbox, 0)
            cur_frame_track_ids.append(new_track_id)

        # Track incoming frames
        for frame_id in range(1, min(self.track_steps, len(bboxes_seq))):
            prev_bboxes = bboxes_seq[frame_id - 1]
            cur_bboxes = bboxes_seq[frame_id]
            if scores_seq is not None:
                prev_bboxes = prev_bboxes[scores_seq[frame_id - 1] >= self.min_score]
                cur_bboxes = cur_bboxes[scores_seq[frame_id] >= self.min_score]
            # TODO: improvement: velocity prediction based cost matrix
            predicted_velocities = self.velocity_prediction(cur_frame_track_ids)
            predicted_cur_bboxes = self.bbox_prediction(predicted_velocities, prev_bboxes)
            # TODO: improvement: uncomment when applying motion feature
            assign_matrix, cost_matrix = self.track_consecutive_frame(
                prev_bboxes, cur_bboxes#, predicted_cur_bboxes
            )
            # TODO: uncomment if check association similarity rates between the two algorithm
            # total_assign_diff.append(assign_diff)

            prev_frame_track_ids = deepcopy(cur_frame_track_ids)
            cur_frame_track_ids = []
            prev_bbox_ids, cur_bbox_ids = np.where(assign_matrix)
            for j in range(cur_bboxes.shape[0]):
                if j in cur_bbox_ids:
                    i = prev_bbox_ids[cur_bbox_ids == j]
                    assert len(i) == 1
                    i = i.item()
                    track_id = prev_frame_track_ids[i]
                    self.tracks[track_id].insert_new_observation(
                        frame_id, cur_bboxes[j], cost_matrix[i, j]
                    )
                else:
                    track_id = self.create_new_tracklet(frame_id, cur_bboxes[j], 0)
                cur_frame_track_ids.append(track_id)
        # Uncomment if check association similarity rates between the two algorithm
        # print("Avg assignment difference (greedy and Hung) rate: ", sum(total_assign_diff) / len(total_assign_diff))