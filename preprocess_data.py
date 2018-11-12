#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import numpy as np
import pandas as pd
import scipy.io
import cv2

def resize(im, desired_size = None):
	old_size = im.shape[:2] # old_size is in (height, width) format

	ratio = float(desired_size)/max(old_size)
	new_size = tuple([int(x*ratio) for x in old_size])

	# new_size should be in (width, height) format

	im = cv2.resize(im, (new_size[1], new_size[0]))

	delta_w = desired_size - new_size[1]
	delta_h = desired_size - new_size[0]
	top, bottom = delta_h//2, delta_h-(delta_h//2)
	left, right = delta_w//2, delta_w-(delta_w//2)

	color = [0, 0, 0]
	new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
		value=color)
	return new_im


def convert_pose(vect):
	M, _ = cv2.Rodrigues(np.array(vect).astype(np.float32))
	vec = M[:, 2]
	yaw = np.arctan2(vec[0], vec[2])
	pitch = np.arcsin(vec[1])
	return np.array([yaw, pitch])


def convert_gaze(vect):
	x, y, z = vect
	yaw = np.arctan2(-x, -z)
	pitch = np.arcsin(-y)
	return np.array([yaw, pitch])

def get_eval_info(subject_id, evaldir):
	df = pd.read_csv(
		os.path.join(evaldir, '{}.txt'.format(subject_id)),
		delimiter=' ',
		header=None,
		names=['path', 'side'])
	df['day'] = df.path.apply(lambda path: path.split('/')[0])
	df['filename'] = df.path.apply(lambda path: path.split('/')[1])
	df = df.drop(['path'], axis=1)
	return df


def get_subject_data(subject_id, datadir, evaldir):
	left_images = {}
	left_poses = {}
	left_gazes = {}
	right_images = {}
	right_poses = {}
	right_gazes = {}
	filenames = {}
	dirpath = os.path.join(datadir, subject_id)
	for name in sorted(os.listdir(dirpath)):
		path = os.path.join(dirpath, name)
		matdata = scipy.io.loadmat(
			path, struct_as_record=False, squeeze_me=True)
		data = matdata['data']

		# print type(matdata)
		# print matdata
		# raise "debug"

		day = os.path.splitext(name)[0]
		left_images[day] = data.left.image
		left_poses[day] = data.left.pose
		left_gazes[day] = data.left.gaze

		# print ("data.left.image.shape")
		# print (data.left.image.shape)
		# raise ("debug")

		right_images[day] = data.right.image
		right_poses[day] = data.right.pose
		right_gazes[day] = data.right.gaze

		filenames[day] = matdata['filenames']

		if not isinstance(filenames[day], np.ndarray):
			left_images[day] = np.array([left_images[day]])
			left_poses[day] = np.array([left_poses[day]])
			left_gazes[day] = np.array([left_gazes[day]])
			right_images[day] = np.array([right_images[day]])
			right_poses[day] = np.array([right_poses[day]])
			right_gazes[day] = np.array([right_gazes[day]])
			filenames[day] = np.array([filenames[day]])

	images = []
	poses = []
	gazes = []

	# left_images_list = []
	# left_poses_list = []
	# left_gazes_list = []
	#
	# right_images_list = []
	# right_poses_list = []
	# right_gazes_list = []

	df = get_eval_info(subject_id, evaldir)
	for _, row in df.iterrows():
		day = row.day
		index = np.where(filenames[day] == row.filename)[0][0]

		if row.side == 'left':
			image = left_images[day][index]
			scaled_img = resize(image, 227)
			rgb_img = cv2.cvtColor(scaled_img,cv2.COLOR_GRAY2RGB)

			pose = convert_pose(left_poses[day][index])
			gaze = convert_gaze(left_gazes[day][index])

			# left_images_list.append(rgb_img)
			# left_poses_list.append(pose)
			# left_gazes_list.append(gaze)

		else:
			image = right_images[day][index][:, ::-1]
			# cv2.imshow("image", image)

			# scaled_img = cv2.resize(image, (64, 64), interpolation=interp)
			scaled_img = resize(image, 227)

			# print("scaled_img.shape: ", scaled_img.shape)

			# cv2.imshow("scaled", scaled_img)

			rgb_img = cv2.cvtColor(scaled_img,cv2.COLOR_GRAY2RGB)

			# print("rgb_img.shape: ", rgb_img.shape)

			# cv2.imshow("rgb", rgb_img)
			# cv2.waitKey(0)

			# print ("right_poses[day][index]): ", right_poses[day][index])
			pose = convert_pose(right_poses[day][index]) * np.array([-1, 1])
			# print ("pose.shape: ", pose.shape)
			# print ("pose: ", pose)

			# print ("right_gazes[day][index]): ", right_gazes[day][index])
			gaze = convert_gaze(right_gazes[day][index]) * np.array([-1, 1])
			# print ("gaze.shape: ", gaze.shape)
			# print ("gaze: ", gaze)

# ----------------------------------------------------------------
#
# 			h, w, _= rgb_img.shape
# 			increase = 5
# 			rgb_img = cv2.resize(rgb_img, (h * increase, w * increase))
#
#
# 			print gaze
# 			pitch, yaw = gaze          # theta, alph
# 			increase = 100
#
# 			# y_x, y_y = int(pitch * increase), int(yaw * increase)
# 			# y_x, y_y = -int(pitch * increase), int(yaw * increase)
# 			# y_x, y_y = int(pitch * increase), -int(yaw * increase)
#
# 			# *****
# 			y_x1, y_y1 = -int(pitch * increase), -int(yaw * increase) # correct option !!!!!!
# 			# *****
#
# 			# y_x, y_y = int(yaw * increase), int(pitch * increase)
# 			# y_x, y_y = -int(yaw * increase), int(pitch * increase)
# 			# y_x, y_y = int(yaw * increase), -int(pitch * increase)
# 			# y_x2, y_y2 = -int(yaw * increase), -int(pitch * increase)
#
# 			# print y_x, y_y
#
# 			h, w, _= rgb_img.shape
#
# 			cx, cy = w/2.0, h/2.0
# 			cv2.circle(rgb_img,(int(cx), int(cy)), 5, (0,0,255), -1)
# 			cv2.line(rgb_img, (int(cx), int(cy)), (int(cx + y_x1), int(cy + y_y1)), (255, 0, 0), 3)
#
# 			# cv2.line(rgb_img, (int(cx), int(cy)), (int(cx + y_x2), int(cy + y_y2)), (0, 255, 0), 3)
#
# 			cv2.imshow("rgb", rgb_img)
# 			cv2.waitKey(0)
# ----------------------------------------------------------------


			# right_images_list.append(rgb_img)
			# right_poses_list.append(pose)
			# right_gazes_list.append(gaze)

		# images.append(image)
		images.append(rgb_img)
		poses.append(pose)
		gazes.append(gaze)

	images = np.array(images).astype(np.float32) / 255
	poses = np.array(poses).astype(np.float32)
	gazes = np.array(gazes).astype(np.float32)

	# left_images_list = np.array(left_images_list).astype(np.float32) / 255
	# left_poses_list = np.array(left_poses_list).astype(np.float32)
	# left_gazes_list = np.array(left_gazes_list).astype(np.float32)
	#
	# right_images_list = np.array(right_images_list).astype(np.float32) / 255
	# right_poses_list = np.array(right_poses_list).astype(np.float32)
	# right_gazes_list = np.array(right_gazes_list).astype(np.float32)

	return images, poses, gazes
	# return left_images_list, left_poses_list, left_gazes_list, right_images_list, right_poses_list, right_gazes_list


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', type=str, required=True)
	parser.add_argument('--outdir', type=str, required=True)
	args = parser.parse_args()

	outdir = args.outdir
	if not os.path.exists(outdir):
		os.makedirs(outdir)

	left_images_total = []
	left_poses_total = []
	left_gazes_total = []

	right_images_total = []
	right_poses_total = []
	right_gazes_total = []

	for subject_id in range(15):
		subject_id = 'p{:02}'.format(subject_id)

		print ("subject_id: ", subject_id)

		datadir = os.path.join(args.dataset, 'Data', 'Normalized')
		# datadir = os.path.join(args.dataset, 'Data', 'Original')

		evaldir = os.path.join(args.dataset, 'Evaluation Subset',
							   'sample list for eye image')

		images, poses, gazes = get_subject_data(subject_id, datadir, evaldir)
		# left_images, left_poses, left_gazes, right_images, right_poses, right_gazes = get_subject_data(subject_id, datadir, evaldir)

		# left_images_total.extend(left_images)
		# left_poses_total.extend(left_poses)
		# left_gazes_total.extend(left_gazes)
		#
		# right_images_total.extend(right_images)
		# right_poses_total.extend(right_poses)
		# right_gazes_total.extend(right_gazes)

		# outpath = os.path.join(outdir, subject_id)
		#
		# np.savez(outpath, image=images, pose=poses, gaze=gazes)

	# outpath = os.path.join(outdir, "rgb_227")
	# np.savez(outpath, left_image=left_images_total, left_pose=left_poses_total, left_gaze=left_gazes_total,
	# right_image=right_images_total, right_pose=right_poses_total, right_gaze=right_gazes_total)


if __name__ == '__main__':
	main()
