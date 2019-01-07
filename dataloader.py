# coding: utf-8

import os
import numpy as np
import  cv2
import torch
import torch.utils.data



class MPIIGazeDataset(torch.utils.data.Dataset):
    def __init__(self, subject_id, dataset_dir, if_classifier, additional_ground_truth=None, if_head=False, if_face=False):

        limit = 9999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999
        # limit = 100

        self.if_classifier = if_classifier
        self.if_head = if_head
        self.if_face = if_face
        path = os.path.join(dataset_dir, '{}.npz'.format(subject_id))

        print("---------------------------------------")
        print("loading: ", path)
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

        with np.load(path) as fin:
            self.images = fin['image'][:limit]
                # .astype(np.float32)
            # self.images = self.convert_numpy(self.images)

            self.gazes = fin['gaze'][:limit]
                # .astype(np.float32)
            # self.gazes = self.convert_numpy(self.gazes)

            if self.if_head:
                self.heads = fin['head'][:limit]
                    # .astype(np.float32)
                # self.heads = self.convert_numpy(self.heads)
                print("self.heads.shape: ", self.heads.shape)
            else:
                self.heads = None

            if self.if_face:
                self.faces = fin['face'][:limit]
                    # .astype(np.float32)
            else:
                self.faces = None

            # self.poses = fin['pose'].astype(np.float32)

            if self.if_classifier:
                self.one_hot_gt = fin['gesture'][:limit]
                # self.one_hot_gt = convert_2_onehot(self.one_hot_gt)

        print("len(self.images): ", len(self.images))

        self.length = len(self.images)
        print("path: ", path)
        print("self.length: ", self.length)

        if additional_ground_truth:
            with np.load(additional_ground_truth) as fin:
                self.one_hot_gt = fin['gesture_ground_truth_list'].astype(np.float32)[:limit]

    def convert_numpy(self, array):
        new_array = np.asarray([np.asarray(element.astype(np.float32)) for element in array])
        return new_array

    def convert_rgb(self, image):
        placeholder = np.ones((image.shape[0], image.shape[1], 3))
        image_rgb = np.zeros_like(placeholder)
        image_rgb[:, :, 0] = image
        image_rgb[:, :, 1] = image
        image_rgb[:, :, 2] = image
        return image_rgb

    def __getitem__(self, index):

        shape = self.images[index].shape
        if len(shape) == 2:
            image = self.convert_rgb(self.images[index])
        else:
            image = self.images[index]

        image = torch.Tensor(image.transpose(2, 0, 1).astype(np.float32))
        gaze = torch.Tensor(self.gazes[index].astype(np.float32))

        # if image.shape != torch.Size([3, 227, 227]) or gaze.shape != torch.Size([2]) or head.shape != torch.Size([2]):
        #     print(image.shape, gaze.shape, head.shape, index)
        #     raise TypeError

        if self.if_classifier:
            return image, self.gaze, self.one_hot_gt[index] - 1
        else:
            if self.if_head:
                head = torch.Tensor(self.heads[index].astype(np.float32))
                if self.if_face:
                    face = torch.Tensor(self.faces[index].astype(np.float32))
                    return image, gaze, head, face
                else:
                    return image, gaze, head
            else:
                return image, gaze


    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__


def get_loader(dataset_dir, test_subject_id, batch_size, num_workers, use_gpu, if_classifier=False):
    # assert os.path.exists(dataset_dir)
    assert test_subject_id in range(15)
    # subject_ids = ['p{:02}'.format(index) for index in range(15)]
    subject_ids = ['p{:02}'.format(index) for index in range(2)]


    # test_subject_id = subject_ids[test_subject_id]
    #
    #
    # train_dataset = torch.utils.data.ConcatDataset([
    #     MPIIGazeDataset(subject_id, dataset_dir) for subject_id in subject_ids
    #     if subject_id != test_subject_id
    # ])
    # test_dataset = MPIIGazeDataset(test_subject_id, dataset_dir)


    # train_dataset = MPIIGazeDataset("/home/wangzc/MPIIGaze_Pytorch/gh_exp_data/test_gh_2000_train", dataset_dir)
    # test_dataset = MPIIGazeDataset("/home/wangzc/MPIIGaze_Pytorch/gh_exp_data/test_gh_2000_test", dataset_dir)


    # train_dataset = MPIIGazeDataset("train_A", dataset_dir)
    # test_dataset = MPIIGazeDataset("test_A", dataset_dir)

    # train_dataset = MPIIGazeDataset("train", dataset_dir)
    # test_dataset = MPIIGazeDataset("test", dataset_dir)

    # train_dataset = MPIIGazeDataset("old_data/first_round/train_A", dataset_dir)
    # test_dataset = MPIIGazeDataset("old_data/first_round/test_A", dataset_dir)
    #
    # test_dataset = MPIIGazeDataset("1216_2018_test_A_classifier", dataset_dir)

    # train_dataset = MPIIGazeDataset("1221_2018_train_only_A_classifier", dataset_dir, "1219_2018_train_only_A_classifier_groundtruth.npz")
    # test_dataset = MPIIGazeDataset("1221_2018_test_only_A_classifier", dataset_dir, "1219_2018_test_only_A_classifier_groundtruth.npz")
    #
    # train_dataset = MPIIGazeDataset("1221_2018_train_only_A_classifier", dataset_dir)
    # test_dataset = MPIIGazeDataset("1221_2018_test_only_A_classifier", dataset_dir)

    # train_dataset = MPIIGazeDataset("1221_2018_train_only_A_classifier", dataset_dir, if_classifier, "1221_2018_train_only_A_classifier_groundtruth.npz")
    # test_dataset = MPIIGazeDataset("1221_2018_test_only_A_classifier", dataset_dir, if_classifier, "1221_2018_test_only_A_classifier_groundtruth.npz")

    # train_dataset = MPIIGazeDataset("1221_2018_train_4_camera", dataset_dir, if_classifier)
    # test_dataset = MPIIGazeDataset("1221_2018_test_4_camera", dataset_dir, if_classifier)
    #
    # train_dataset = MPIIGazeDataset("1226_2018_train_4_camera_hog", dataset_dir, if_classifier)
    # test_dataset = MPIIGazeDataset("1226_2018_test_4_camera_hog", dataset_dir, if_classifier)
    #
    # train_dataset = MPIIGazeDataset("1228_2018_train_4_camera_hog_64", dataset_dir, if_classifier)
    # test_dataset = MPIIGazeDataset("1228_2018_test_4_camera_hog_64", dataset_dir, if_classifier)

    # train_dataset = MPIIGazeDataset("0102_2019_train_4_camera_hog", dataset_dir, if_classifier)
    # test_dataset = MPIIGazeDataset("0102_2019_test_4_camera_hog", dataset_dir, if_classifier)
    #
    # train_dataset = MPIIGazeDataset("0103_2019_train_only_A_hog", dataset_dir, if_classifier)
    # test_dataset = MPIIGazeDataset("0103_2019_test_only_A_hog", dataset_dir, if_classifier)

    # train_dataset = MPIIGazeDataset("0105_2019_4_camera_head_11.07_12.06_train", dataset_dir, if_classifier,
    #                                 if_head=True, if_face=False)
    # test_dataset = MPIIGazeDataset("0105_2019_4_camera_head_11.07_12.06_test", dataset_dir, if_classifier,
    #                                if_head=True, if_face=False)

    # train_dataset = MPIIGazeDataset("0107_2019_4_camera_head_face_11.07_12.06_train", dataset_dir, if_classifier,
    #                                 if_head=True, if_face=False)
    # test_dataset = MPIIGazeDataset("0107_2019_4_camera_head_face_11.07_12.06_test", dataset_dir, if_classifier,
    #                                if_head=True, if_face=False)

    train_dataset = MPIIGazeDataset("0107_2019_4_camera_head_face_11.07_12.06_train", dataset_dir, if_classifier,
                                    if_head=True, if_face=True)
    test_dataset = MPIIGazeDataset("0107_2019_4_camera_head_face_11.07_12.06_test", dataset_dir, if_classifier,
                                   if_head=True, if_face=True)


    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_gpu,
        drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=use_gpu,
        drop_last=False,
    )
    return train_loader, test_loader
    # return train_loader, None
    # return None, test_loader

def convert_2_onehot(gesture_ground_truth):
    gesture_ground_truth_list = []

    for index in range(len(gesture_ground_truth)):
        gesture_onehot_list = [0.0] * 10
        gesture = gesture_ground_truth[index]
        gesture_onehot_list[int(gesture - 1)] = 1.0
        gesture_ground_truth_list.append(gesture_onehot_list)

    return np.asarray(gesture_ground_truth_list)

