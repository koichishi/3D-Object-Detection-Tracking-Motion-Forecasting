# read the final result and plot the diagram
import pickle
import os

import numpy as np
import matplotlib.pyplot as plt

from tracking.metrics.evaluator import EvaluationResult

result_path = "<result address needed>"
result_file = "<result filename needed>"

if __name__ == "__main__":
    with open(os.path.join(result_path, result_file), "rb") as f:
        result = pickle.load(f)
    # Round, keep only the mean metrics
    tmp = {}
    for hyp in result:
        tmp[round(hyp, 1)] = result[hyp][0]

    # Split mota, motp, mostly_tracked/lost, partially_tracked into 5 lists
    metrics = []
    metrics.append([score[0] for score in tmp.items()])
    metrics.append([score[1].mota for score in tmp.items()])
    metrics.append([score[1].motp for score in tmp.items()])
    metrics.append([score[1].mostly_tracked for score in tmp.items()])
    metrics.append([score[1].mostly_lost for score in tmp.items()])
    metrics.append([score[1].partially_tracked for score in tmp.items()])
    metrics = np.asarray(metrics)

    plt.plot(metrics[0], metrics[1], label = "MOTA")
    plt.plot(metrics[0], metrics[2], label = "MOTP")
    plt.plot(metrics[0], metrics[3], label = "Mostly Tracked")
    plt.plot(metrics[0], metrics[4], label = "Mostly Lost")
    plt.plot(metrics[0], metrics[5], label = "Partially Tracked")
    plt.xticks(np.arange(0, 3, 0.2))
    plt.legend()
    plt.show()
    plt.savefig(
        os.path.join(result_path, f"hyper_param_plot.png"),
    )


    
    