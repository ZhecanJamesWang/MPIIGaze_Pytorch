import numpy as np

# with np.load(path) as fin:
#     self.images = fin['image'].astype(np.float32)
#     # self.poses = fin['pose'].astype(np.float32)
#     self.gazes = fin['gaze'].astype(np.float32)
#     # self.heads = fin['head'].astype(np.float32)

# path = "1219_2018_test_only_A_classifier_one_hot_groundtruth.npz"
# path = "1219_2018_test_only_A_classifier_one_hot_groundtruth.npz"
#
# with np.load(path) as fin:
#     one_hot_gt = fin['gesture_ground_truth_list'].astype(np.float32)[:100]



# path = "1221_2018_train_only_A_classifier.npz"
path = "1221_2018_train_only_A_classifier_groundtruth.npz"

with np.load(path) as fin:
    # one_hot_gt = fin['gesture'].astype(np.float32)
    one_hot_gt = fin['gesture_ground_truth_list'].astype(np.float32)

    print(one_hot_gt[:100])
    print(max(one_hot_gt))
    print(min(one_hot_gt))
