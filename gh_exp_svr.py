
import numpy as np

# f = open('gh_exp_data/features.txt', "r")
# lines = f.readlines()
#
# print("len(lines): ", len(lines))
#
# head_features = []
# counter = 0
# for line in lines:
#     feature = []
#     parts = line.split(" ")
#     for part in parts:
#         feature.append(float(part))
#     feature = np.asarray(feature)
#     head_features.append(feature)
#     counter += 1
#
#     if counter == 20000:
#         break
#
#
# head_features = np.asarray(head_features)
#
# print(head_features.shape)




import numpy as np
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor


def compute_angle_error(preds, labels):
    err = np.mean(abs(preds - labels))

    return err * 180 / np.pi



file = 'gh_exp_data/gh_exp_features.npz'

data = np.load(file)

# length = 1000
length = 15000
image_list = data['image_list'][:length]
gaze_list = data['gaze_list'][:length]
gaze_features = data['gaze_features'][:length]
head_features = data['head_features'][:length]

print("image_list.shape: ", image_list.shape)
print("gaze_list.shape: ", gaze_list.shape)
print("gaze_features.shape: ", gaze_features.shape)
print("head_features.shape: ", head_features.shape)



features = np.concatenate((gaze_features, head_features), axis=1)


print("features.shape: ", features.shape)

cut = int(length * 0.95)
X_train = features[:cut]
X_test = features[cut:]
y_train = gaze_list[:cut]
y_test = gaze_list[cut:]


max_depth = 100
clf = MultiOutputRegressor(RandomForestRegressor(n_estimators=1000,
                                                          max_depth=max_depth,
                                                          random_state=0))

# clf = SVR(kernel='rbf', C=1e3, gamma=0.1)

clf.fit(X_train, y_train)


y_predict = clf.predict(X_test)


print("y_predict.shape: ", y_predict.shape)

print(compute_angle_error(y_predict, y_test))

# print(y_test[:10])
# print(y_predict[:10])
#
# confidence = clf.score(X_test, y_test)
# print(confidence)
