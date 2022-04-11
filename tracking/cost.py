import numpy as np
from shapely.geometry import Polygon

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
            xw1, xh1, yw1, yh1 = 0.5*bboxes1[i1][2]*np.cos(bboxes1[i1][4]), 0.5*bboxes1[i1][3]*np.sin(bboxes1[i1][4]), 0.5*bboxes1[i1][2]*np.sin(bboxes1[i1][4]), 0.5*bboxes1[i1][3]*np.cos(bboxes1[i1][4])
            xw2, xh2, yw2, yh2 = 0.5*bboxes2[i2][2]*np.cos(bboxes2[i2][4]), 0.5*bboxes2[i2][3]*np.sin(bboxes2[i2][4]), 0.5*bboxes2[i2][2]*np.sin(bboxes2[i2][4]), 0.5*bboxes2[i2][3]*np.cos(bboxes2[i2][4])
            cx1, cy1, cx2, cy2 = bboxes1[i1][0], bboxes1[i1][1], bboxes2[i2][0], bboxes2[i2][1]
            b1 = Polygon([(cx1-xw1+xh1,cy1-yw1-yh1), (cx1-xw1-xh1,cy1-yw1+yh1), (cx1+xw1-xh1,cy1+yw1+yh1), (cx1+xw1+xh1,cy1+yw1-yh1)])
            b2 = Polygon([(cx2-xw2+xh2,cy2-yw2-yh2), (cx2-xw2-xh2,cy2-yw2+yh2), (cx2+xw2-xh2,cy2+yw2+yh2), (cx2+xw2+xh2,cy2+yw2-yh2)])
            a_overlap = b1.intersection(b2).area
            a_union = b1.union(b2).area
            iou_mat[i1][i2] = a_overlap/a_union

    return iou_mat
