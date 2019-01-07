import numpy as np


# file = 'gh_exp_data/gh_exp_features.npz'
file = 'gh_exp_data/test_gh_2000.npz'
data = np.load(file)

image = data['image']
gaze = data['gaze']
head = data['head']

print("image.shape: ", image.shape)
print("gaze.shape: ", gaze.shape)
print("head.shape: ", head.shape)

length = len(image)

cut = int(length * 0.9)
image_train = image[:cut]
image_test = image[cut:]
gaze_train = gaze[:cut]
gaze_test = gaze[cut:]
head_train = head[:cut]
head_test = head[cut:]


print(image_train.shape)
print(image_test.shape)

np.savez("test_gh_2000_train", image=image_train, gaze=gaze_train, head=head_train)
np.savez("test_gh_2000_test", image=image_test, gaze=gaze_test, head=head_test)
