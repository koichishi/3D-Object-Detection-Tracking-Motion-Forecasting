import numpy as np
import torch
from shapely.geometry import Polygon
from detection.types import Detections

def Ciou_2d(bboxes1: np.ndarray, bboxes2: np.ndarray) -> np.ndarray:
    """Computes 2D intersection over union of two sets of bounding boxes

    Args:
        bboxes1: bounding box set of shape [M, 5], each row corresponding to x, y, l, w, yaw of the bounding box
        bboxes2: bounding box set of shape [N, 5], each row corresponding to x, y, l, w, yaw of the bounding box
    Returns:
        iou_mat: matrix of shape [M, N], where iou_mat[i, j] is the 2D IoU value between bboxes[i] and bboxes[j].
        You should use the Polygon class from the shapely package to compute the area of intersection/union.
    """

    M, N = bboxes1.shape[0], bboxes2.shape[0]
    # TODO: Replace this stub code.
    iou_mat = np.zeros((M, N))
    penalty_mat = np.zeros((M, N))
    for i1 in range(M):
        for i2 in range(N):
            x1, y1, w1, l1, yaw1 = bboxes1[i1][0], bboxes1[i1][1], bboxes1[i1][3], bboxes1[i1][2], bboxes1[i1][4]
            x2, y2, w2, l2, yaw2 = bboxes2[i2][0], bboxes2[i2][1], bboxes2[i2][3], bboxes2[i2][2], bboxes2[i2][4]
            xw1, xh1, yw1, yh1 = 0.5*l1*np.cos(yaw1), 0.5*w1*np.sin(yaw1), 0.5*l1*np.sin(yaw1), 0.5*w1*np.cos(yaw1)
            xw2, xh2, yw2, yh2 = 0.5*l2*np.cos(yaw2), 0.5*w2*np.sin(yaw2), 0.5*l2*np.sin(yaw2), 0.5*w2*np.cos(yaw2)
            b1 = Polygon([
                (x1-xw1+xh1,y1-yw1-yh1), 
                (x1-xw1-xh1,y1-yw1+yh1), 
                (x1+xw1-xh1,y1+yw1+yh1), 
                (x1+xw1+xh1,y1+yw1-yh1)]
                )
            b2 = Polygon([
                (x2-xw2+xh2,y2-yw2-yh2), 
                (x2-xw2-xh2,y2-yw2+yh2), 
                (x2+xw2-xh2,y2+yw2+yh2), 
                (x2+xw2+xh2,y2+yw2-yh2)]
                )
            a_overlap = b1.intersection(b2).area
            a_union = b1.union(b2).area
            iou = a_overlap/a_union
            iou_mat[i1][i2] = iou

            xmin, ymin, xmax, ymax = b1.union(b2).bounds
            euc_dist2 = (x1-x2) ** 2 + (y1-y2) ** 2
            # Diagonal length of smallest enclose box that covers bbox1 and bbox2
            c = np.sqrt((xmax - xmin) ** 2 + (ymax - ymin) ** 2) 
            # Consistancy of aspect ratio
            v = 4.0 / np.pi ** 2 * (np.arctan(w1 / l1) - np.arctan(w2 / l2)) ** 2
            # trade-off param
            alpha = v / ((1 - iou) + v)
            penalty_mat[i1][i2] = euc_dist2 / c ** 2 + alpha * v

    return iou_mat - penalty_mat


def iou_2d(bboxes1: np.ndarray, bboxes2: np.ndarray) -> np.ndarray:
    """Computes 2D intersection over union of two sets of bounding boxes

    Args:
        bboxes1: bounding box set of shape [M, 5], each row corresponding to x, y, l, w, yaw of the bounding box
        bboxes2: bounding box set of shape [N, 5], each row corresponding to x, y, l, w, yaw of the bounding box
    Returns:
        iou_mat: matrix of shape [M, N], where iou_mat[i, j] is the 2D IoU value between bboxes[i] and bboxes[j].
        You should use the Polygon class from the shapely package to compute the area of intersection/union.
    """
    M, N = bboxes1.shape[0], bboxes2.shape[0]
    # TODO: Replace this stub code.
    iou_mat = np.zeros((M, N))
    for i1 in range(M):
        for i2 in range(N):
            x1, y1, w1, l1, yaw1 = bboxes1[i1][0], bboxes1[i1][1], bboxes1[i1][3], bboxes1[i1][2], bboxes1[i1][4]
            x2, y2, w2, l2, yaw2 = bboxes2[i2][0], bboxes2[i2][1], bboxes2[i2][3], bboxes2[i2][2], bboxes2[i2][4]
            xw1, xh1, yw1, yh1 = 0.5*l1*np.cos(yaw1), 0.5*w1*np.sin(yaw1), 0.5*l1*np.sin(yaw1), 0.5*w1*np.cos(yaw1)
            xw2, xh2, yw2, yh2 = 0.5*l2*np.cos(yaw2), 0.5*w2*np.sin(yaw2), 0.5*l2*np.sin(yaw2), 0.5*w2*np.cos(yaw2)
            b1 = Polygon([
                (x1-xw1+xh1,y1-yw1-yh1), 
                (x1-xw1-xh1,y1-yw1+yh1), 
                (x1+xw1-xh1,y1+yw1+yh1), 
                (x1+xw1+xh1,y1+yw1-yh1)]
                )
            b2 = Polygon([
                (x2-xw2+xh2,y2-yw2-yh2), 
                (x2-xw2-xh2,y2-yw2+yh2), 
                (x2+xw2-xh2,y2+yw2+yh2), 
                (x2+xw2+xh2,y2+yw2-yh2)]
                )
            a_overlap = b1.intersection(b2).area
            a_union = b1.union(b2).area
            iou_mat[i1][i2] = a_overlap/a_union

    return iou_mat
