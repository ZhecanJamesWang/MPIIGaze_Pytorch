import numpy as np
#
# file = '1219_2018_test_only_A_classifier.npz'
file = '1219_2018_train_only_A_classifier.npz'

data = np.load(file)
gesture_ground_truth = data['gesture']

gesture_ground_truth_list = []

for index in range(len(gesture_ground_truth)):
    gesture_onehot_list = [0] * 10
    gesture = gesture_ground_truth[index]
    gesture_onehot_list[gesture - 1] = 1
    gesture_ground_truth_list.append(gesture_onehot_list)

header = file.split(".")[0]
np.savez(header + "_one_hot_groundtruth", gesture_ground_truth_list=gesture_ground_truth_list)



# path = "1219_2018_test_only_A_classifier.npz"
# path = "1219_2018_train_only_A_classifier.npz"
# path = "1221_2018_train_only_A_classifier.npz"
path = "1221_2018_test_only_A_classifier.npz"

with np.load(path) as fin:
    one_hot_gt = fin['gesture'].astype(np.float32)
    one_hot_gt -= 1

header = path.split(".")[0]
np.savez(header + "_groundtruth", gesture_ground_truth_list=one_hot_gt)
