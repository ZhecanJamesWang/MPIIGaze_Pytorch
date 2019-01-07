import numpy as np
# import pickle
# from sklearn.utils.extmath import softmax
from sklearn.metrics import log_loss
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import entropy
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools



# with open('gesture_gaze_reference.pickle', 'rb') as handle:
#     gesture_gaze_reference = pickle.load(handle)
#
# print(gesture_gaze_reference)
#

gesture_gaze_reference = {1: np.asarray([-0.53065168, 0.0820604]), 2: np.asarray([1.13251014, -0.17302723]),
                          3: np.asarray([0.85440895, 0.57813982]), 4: np.asarray([0.16542202,  0.95219552]),
                          6: np.asarray([-0.58084195,  0.07719008]), 7: np.asarray([1.15631547, -0.1634327]),
                          8: np.asarray([0.9015506,  0.53202662]), 9: np.asarray([0.14794189,  0.94518323]),
                          10: np.asarray([1.21398637, -0.04157835])}

def softmax(x):
    e_x = np.exp(x - np.max(x)) # 防止exp()数值溢出
    return e_x / e_x.sum(axis=0)

def cross_entropy_via_scipy(x, y):
    ''' SEE: https://en.wikipedia.org/wiki/Cross_entropy'''
    return entropy(x) + entropy(x, y)

def cross_entropy(predictions, targets, epsilon=1e-12):
    """
    Computes cross entropy between targets (encoded as one-hot vectors)
    and predictions.
    Input: predictions (N, k) ndarray
           targets (N, k) ndarray
    Returns: scalar
    """
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    ce = -np.sum(targets*np.log(predictions+1e-9))/N
    return ce

def compute_angle_error(preds, labels):
    err = np.mean(abs(preds - labels))

    return err * 180 / np.pi

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


# file = '0101_2019_regressor_gesture_ground_truth.npz'
# file = '1221_2018_test_only_A_classifier.npz'
file = "0101_2019_regressor_gestures_list.npz"
data = np.load(file)
gesture_ground_truth = data['gestures_list']


file = '0101_2019_regressor_output_list.npz'
data = np.load(file)
output_list = data['outputs_list']


file = '0101_2019_regressor_gazes_list.npz'
data = np.load(file)
gazes_list = data['gazes_list']

# ///////////////////////////////////////////////////////
#

pred_binary_list = []
gesture_binary_list = []

gesture_ground_truth_list = []
predict_list = []

attention_count = 0
positive_attention_predict_count = 0
false_attention_predict_count = 0

fatigue_count = 0
positive_fatigue_predict_count = 0
false_fatigue_predict_count = 0

class_names = [1, 2, 3, 4, 6, 7, 8, 9, 10]
pred_list = []
gesture_list = []

actual_err_dict = {}

for index in range(len(gesture_ground_truth)):
    gaze = gazes_list[index]

    gesture = int(gesture_ground_truth[index])

    output = output_list[index]

    if gesture != 5:

        actual_err = compute_angle_error(output, gaze)
        key = str(gesture)

        if key not in actual_err_dict:
            actual_err_dict[key] = [actual_err]
        else:
            actual_err_dict[key].append(actual_err)

        gesture_list.append(gesture)
        err_list = []
        gesture_onehot_list = [0] * 9
        for key, value in gesture_gaze_reference.items():
            # print(key, value)
            err = compute_angle_error(output, value)
            err_list.append(err)

        if gesture < 5:
            gesture_onehot_list[gesture - 1] = 1
        else:
            gesture_onehot_list[gesture - 2] = 1

# ///////////////////////////////////////////
        min_err = min(err_list)
        predict = err_list.index(min_err)
        if predict < 4:
            predict += 1
        else:
            predict += 2

        if predict == 10:
            pred_binary_list.append(1)
        else:
            pred_binary_list.append(0)
        #
        if gesture == 10:
            gesture_binary_list.append(1)
            attention_count += 1
            if predict == 10:
                positive_attention_predict_count += 1
            else:
                false_fatigue_predict_count += 1
        else:
            gesture_binary_list.append(0)
            fatigue_count += 1
            if predict != 10:
                positive_fatigue_predict_count += 1
            else:
                false_attention_predict_count += 1
# ///////////////////////////////////////////

        err_list = 1/np.asarray(err_list)
        # print("softmax(err_list): ", softmax(err_list))

        # print("gesture: ", gesture)
        # print("err_list: ", err_list)
        # print("gesture_onehot_list: ", gesture_onehot_list)

        pred_list.append(predict)

        predict_list.append(err_list)
        gesture_ground_truth_list.append(gesture_onehot_list)

# print("gesture_ground_truth_list[:3]: ", gesture_ground_truth_list[:50])
# gesture_ground_truth_list = np.asarray(gesture_ground_truth_list).reshape(-1, 1)
#
# print("gesture_ground_truth_list.shape: ", gesture_ground_truth_list.shape)
#
# gesture_ground_truth_list = enc.fit_transform(gesture_ground_truth_list)
#
# for i in range(50):
#     print(gesture_ground_truth_list[i])

err_rate = sum(abs(np.asarray(pred_binary_list) - np.asarray(gesture_binary_list)))/len(pred_binary_list)
print("cumulative err rate: ", err_rate, "cumulative accuracy rate: ", 1 - err_rate)
print("true positive: ", positive_attention_predict_count)
print("true negative: ", positive_fatigue_predict_count)
print("false positive: ", false_attention_predict_count)
print("false negative: ", false_fatigue_predict_count)
print("positive count: ", attention_count)
print("negative count: ", fatigue_count)

print(cross_entropy(gesture_ground_truth_list, predict_list))
print(cross_entropy_via_scipy(gesture_ground_truth_list, predict_list))

# print(cross_entropy(pred_binary_list, gesture_binary_list))


# Compute confusion matrix
cnf_matrix = confusion_matrix(gesture_list, pred_list)
np.set_printoptions(precision=2)

# # Plot non-normalized confusion matrix
# plt.figure()
# plot_confusion_matrix(cnf_matrix, classes=class_names,
#                       title='Confusion matrix, without normalization')
#
# plt.savefig('Confusion matrix, without normalization_0101_2019_regressor')
#
# # Plot normalized confusion matrix
# plt.figure()
# plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
#                       title='Normalized confusion matrix')
#
# plt.savefig('Normalized confusion matrix_0101_2019_regressor')
# # plt.show()


counter = 0
for key, value in actual_err_dict.items():
    plt.figure(counter)
    counter += 1
    plt.hist(value, bins=30)
    # normed = True,
    plt.title(key)
    plt.xlabel("Difference(degree)")
    plt.ylabel("Frequency")
    print(key)
    plt.savefig(key)
