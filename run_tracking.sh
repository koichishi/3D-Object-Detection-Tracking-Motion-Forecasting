srun -p csc490-compute -c 6 -N 1 --gres gpu -o /h/u9/c7/00/shigongy/Desktop/csc490/3D-Object-Detection/run_outfile 
# hung track imp on valid
# track
python3 -m tracking.main track --dataset_path=/u/csc490h/dataset --tracker_associate_method=hungarian --detection_path=/h/u9/c7/00/shigongy/Desktop/csc490/3D-Object-Detection/tracking/detection_results/csc490_detector --result_path=/h/u9/c7/00/shigongy/Desktop/csc490/3D-Object-Detection/tracking/tracking_results/hungarian_val_imp/results.pkl
# eval
python3 -m tracking.main evaluate --result_path=/h/u9/c7/00/shigongy/Desktop/csc490/3D-Object-Detection/tracking/tracking_results/hungarian_val_imp/results.pkl
# vis
python3 -m tracking.main visualize --result_path=/h/u9/c7/00/shigongy/Desktop/csc490/3D-Object-Detection/tracking/tracking_results/hungarian_val_imp/results.pkl
# hyp tuning on train
python3 -m tracking.main hyperparameter_tuning --dataset_path=/u/csc490h/dataset --tracker_associate_method=hungarian --detection_path=/h/u9/c7/00/shigongy/Desktop/csc490/3D-Object-Detection/tracking/detection_results/csc490_detector_train --result_path=/h/u9/c7/00/shigongy/Desktop/csc490/3D-Object-Detection/tracking/tracking_results/hungarian_train_imp/

# hung track on valid
# track
python3 -m tracking.main track --dataset_path=/u/csc490h/dataset --tracker_associate_method=hungarian --detection_path=/h/u9/c7/00/shigongy/Desktop/csc490/3D-Object-Detection/tracking/detection_results/csc490_detector --result_path=/h/u9/c7/00/shigongy/Desktop/csc490/3D-Object-Detection/tracking/tracking_results/hungarian_val/results.pkl
# eval
python3 -m tracking.main evaluate --result_path=/h/u9/c7/00/shigongy/Desktop/csc490/3D-Object-Detection/tracking/tracking_results/hungarian_val/results.pkl
# vis
python3 -m tracking.main visualize --result_path=/h/u9/c7/00/shigongy/Desktop/csc490/3D-Object-Detection/tracking/tracking_results/hungarian_val/results.pkl
