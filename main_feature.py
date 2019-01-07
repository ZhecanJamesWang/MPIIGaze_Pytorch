#!/usr/bin/env python
# coding: utf-8
import os
import time
import json
from collections import OrderedDict
import importlib
import logging
import argparse
import numpy as np
import random
import cv2
import datetime

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torch.backends.cudnn
import torchvision.utils

# os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
# os.environ["CUDA_VISIBLE_DEVICES"]="2"
# os.environ["CUDA_VISIBLE_DEVICES"]="3"
# os.environ["CUDA_VISIBLE_DEVICES"]="4"
# os.environ["CUDA_VISIBLE_DEVICES"]="5"

# try:
from tensorboardX import SummaryWriter
is_tensorboard_available = True
# except Exception:
#     is_tensorboard_available = False

from dataloader import get_loader

# torch.backends.cudnn.benchmark = False

logging.basicConfig(
    format='[%(asctime)s %(name)s %(levelname)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.DEBUG)
logger = logging.getLogger(__name__)

global_step = 0

now = datetime.datetime.now()
date = now.strftime("%Y-%m-%d-%H-%M")

record_file_name = date + '_record.txt'
records = ""
records_count = 0


def str2bool(s):
    if s.lower() == 'true':
        return True
    elif s.lower() == 'false':
        return False
    else:
        raise RuntimeError('Boolean value expected')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--arch', type=str, required=True, choices=['lenet', 'resnet_preact', 'alexnet', 'resnet101', 'resnet34',
                                                    'resnet18', 'resnet10', 'resnet18_gh_exp_1', "resnet34_gh_exp_1",
                                                    "resnet34_gh_exp_2", "resnet34_gh_exp_3"])
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--test_id', type=int, required=True)
    parser.add_argument('--outdir', type=str, required=True)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--num_workers', type=int, default=7)

    # optimizer
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=300)
    parser.add_argument('--base_lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--nesterov', type=str2bool, default=True)
    parser.add_argument('--milestones', type=str, default='[20, 30]')
    parser.add_argument('--lr_decay', type=float, default=0.1)

    # TensorBoard
    parser.add_argument(
        '--tensorboard', dest='tensorboard', action='store_true', default=True)
    parser.add_argument(
        '--no-tensorboard', dest='tensorboard', action='store_false')
    parser.add_argument('--tensorboard_images', action='store_true')
    parser.add_argument('--tensorboard_parameters', action='store_true')

    parser.add_argument('--pretrained_path', type=str, default = "")
    parser.add_argument('--gpu', type=str, required=True)

    args = parser.parse_args()
    # if not is_tensorboard_available:
    #     args.tensorboard = False
    #     args.tensorboard_images = False
    #     args.tensorboard_parameters = False

    args.tensorboard = True
    args.tensorboard_images = True
    args.tensorboard_parameters = True

    assert os.path.exists(args.dataset)
    args.milestones = json.loads(args.milestones)

    return args

args = parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu.strip()

print ("print (torch.cuda.current_device()): ", torch.cuda.current_device())

args.outdir = "results/" + date + "_" + args.outdir


def write_to_file(file_name, content):

	fh = open(file_name, "a")
	fh.write(content)
	fh.close

	content = ""
	return content


def save_to_record(content):
    global records
    global records_count

    # print(content)
    records += content
    records_count += 1

    file_name = args.outdir + "/" + record_file_name

    if records_count % 20 == 0:
        write_to_file(file_name, records)
        records = ""


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, num):
        self.val = val
        self.sum += val * num
        self.count += num
        self.avg = self.sum / self.count

# If we define pitch=0 as horizontal (z=0) and yaw as counter-clockwise from the x axis, then the direction vector will be

# x = cos(yaw)*cos(pitch)
# y = sin(yaw)*cos(pitch)
# z = sin(pitch)


# (pitch, yaw)

def convert_to_unit_vector(angles):
    x = -torch.cos(angles[:, 0]) * torch.sin(angles[:, 1])
    y = -torch.sin(angles[:, 0])
    # z = -torch.cos(angles[:, 1]) * torch.cos(angles[:, 1])
    z = -torch.cos(angles[:, 0]) * torch.cos(angles[:, 1])

    norm = torch.sqrt(x**2 + y**2 + z**2)
    x /= norm
    y /= norm
    z /= norm
    return x, y, z


# def compute_angle_error(preds, labels):
#     pred_x, pred_y, pred_z = convert_to_unit_vector(preds)
#     label_x, label_y, label_z = convert_to_unit_vector(labels)
#     angles = pred_x * label_x + pred_y * label_y + pred_z * label_z
#     return torch.acos(angles) * 180 / np.pi

def compute_angle_error(preds, labels):
    err = torch.abs(preds - labels).mean()

    return err * 180 / np.pi

def train(epoch, model, optimizer, criterion, train_loader, config, writer, if_gaze = True):
    global global_step

    logger.info('Train {}'.format(epoch))
    save_to_record('Train {}'.format(epoch) + "\n")

    model.train()

    # if if_gaze:
    loss_meter = AverageMeter()
    angle_error_meter = AverageMeter()
    start = time.time()
    # gaze_features = []
    # head_features = []
    # gazes_list = []
    # image_list = []

    # for step, (images, poses, gazes) in enumerate(train_loader):
    # for step, (images, gazes) in enumerate(train_loader):
    for step, (images, gazes, heads) in enumerate(train_loader):
        global_step += 1

        # print ("before scale")
        # print (images.shape)

        if config['tensorboard_images'] and step == 0:
            image = torchvision.utils.make_grid(
                images, normalize=True, scale_each=True)
            writer.add_image('Train/Image', image, epoch)

        # print ("after scale")
        # print (images.shape)
        images = images.cuda()
        # poses = poses.cuda()
        gazes = gazes.cuda()
        heads = heads.cuda()

        # poses = poses.float()
        # gazes = gazes.float()

        optimizer.zero_grad()

        # outputs = model(images, poses)
        # outputs = model(images)
        outputs = model(images, heads)

        # print("outputs.shape: ", outputs.shape)
        # print("heads.shape: ", heads.shape)

        # images = images.detach().cpu().numpy()
        # gazes = gazes.detach().cpu().numpy()
        # gaze_feature = outputs.detach().cpu().numpy()
        # head_feature = heads.detach().cpu().numpy()
        #
        # image_list.extend(images)
        # gazes_list.extend(gazes)
        # gaze_features.extend(gaze_feature)
        # head_features.extend(head_feature)

        loss = criterion(outputs, gazes)
        loss.backward()

        optimizer.step()

        angle_error = compute_angle_error(outputs, gazes).mean()

        # num = images.size(0)
        num = 1
        loss_meter.update(loss.item(), num)
        angle_error_meter.update(angle_error.item(), num)

        if config['tensorboard']:
            writer.add_scalar('Train/RunningLoss', loss_meter.val, global_step)


        if step % 10 == 0:

            logger.info('Epoch {} Step {}/{} '
                        'Loss {:.4f} ({:.4f}) '
                        'AngleError {:.2f} ({:.2f})'.format(
                            epoch,
                            step,
                            len(train_loader),
                            loss_meter.val,
                            loss_meter.avg,
                            angle_error_meter.val,
                            angle_error_meter.avg,
                        ))
            save_to_record('Epoch {} Step {}/{} '
                        'Loss {:.4f} ({:.4f}) '
                        'AngleError {:.2f} ({:.2f})'.format(
                            epoch,
                            step,
                            len(train_loader),
                            loss_meter.val,
                            loss_meter.avg,
                            angle_error_meter.val,
                            angle_error_meter.avg,
                        ) + "\n")
        if step % 10 == 0:
            logger.info(json.dumps(vars(args), indent=2))

    # image_list = np.asarray(image_list)
    # gaze_list = np.asarray(gazes_list)
    # gaze_features = np.asarray(gaze_features)
    # head_features = np.asarray(head_features)
    #
    # print("image_list.shape: ", image_list.shape)
    # print("gaze_list.shape: ", gaze_list.shape)
    # print("gaze_features.shape: ", gaze_features.shape)
    # print("head_features.shape: ", head_features.shape)
    #
    # np.savez("gh_exp_features", image_list=image_list, gaze_list=gaze_list, gaze_features=gaze_features, head_features=head_features)
    #
    # raise("debug")

    elapsed = time.time() - start
    logger.info('Elapsed {:.2f}'.format(elapsed))
    save_to_record('Elapsed {:.2f}'.format(elapsed) + "\n")

    outdir = args.outdir
    model_path = os.path.join(outdir, 'model_state.pth')
    torch.save(model.state_dict(), model_path)

    if config['tensorboard']:
        writer.add_scalar('Train/Loss', loss_meter.avg, epoch)
        writer.add_scalar('Train/AngleError', angle_error_meter.avg, epoch)
        writer.add_scalar('Train/Time', elapsed, epoch)


def test(epoch, model, criterion, test_loader, config, writer):
    logger.info('Test {}'.format(epoch))
    save_to_record('Test {}'.format(epoch) + "\n")

    logger.info(json.dumps(vars(args), indent=2))

    model.eval()

    loss_meter = AverageMeter()
    angle_error_meter = AverageMeter()
    start = time.time()
    # for step, (images, poses, gazes) in enumerate(test_loader):
    # for step, (images, gazes) in enumerate(test_loader):
    for step, (images, gazes, heads) in enumerate(test_loader):

        # print ("images.shape: ", images.shape)

        if config['tensorboard_images'] and epoch == 0 and step == 0:
            image = torchvision.utils.make_grid(
                images, normalize=True, scale_each=True)
            writer.add_image('Test/Image', image, epoch)

        images = images.cuda()
        # poses = poses.cuda()
        gazes = gazes.cuda()
        heads = heads.cuda()

        with torch.no_grad():
            # outputs = model(images, poses)
            outputs = model(images, heads)

        # for index in range(len(images)):
        #     image = images[index]
        #     cv2.imshow("image", image)

        loss = criterion(outputs, gazes)

        angle_error = compute_angle_error(outputs, gazes).mean()

        # num = images.size(0)
        num = 1
        loss_meter.update(loss.item(), num)
        angle_error_meter.update(angle_error.item(), num)

    logger.info('Epoch {} Loss {:.4f} AngleError {:.2f}'.format(
        epoch, loss_meter.avg, angle_error_meter.avg))

    save_to_record('Epoch {} Loss {:.4f} AngleError {:.2f}'.format(
        epoch, loss_meter.avg, angle_error_meter.avg) + "\n")


    elapsed = time.time() - start
    logger.info('Elapsed {:.2f}'.format(elapsed))
    save_to_record('Elapsed {:.2f}'.format(elapsed) + "\n")

    if config['tensorboard']:
        if epoch > 0:
            writer.add_scalar('Test/Loss', loss_meter.avg, epoch)
            writer.add_scalar('Test/AngleError', angle_error_meter.avg, epoch)
        writer.add_scalar('Test/Time', elapsed, epoch)

    if config['tensorboard_parameters']:
        for name, param in model.named_parameters():
            writer.add_histogram(name, param, global_step)

    return angle_error_meter.avg

def plot_gaze_pose(center_pt, gaze, pose, image):

    [cx, cy] = center_pt
    [left_yaw, left_pitch] = gaze
    [head_yaw, head_pitch] = pose
    increase = 30

    # y_x, y_y = - np.sin(left_yaw), - np.sin(left_pitch)
    y_x, y_y = - np.sin(head_yaw * np.pi/180), np.sin(head_pitch * np.pi/180)
    # y_x, y_y = left_eye_vector_unit
    print (image)

    print ("cx, cy: ", cx, cy)
    print ("y_x, y_y: ", y_x, y_y)

    y_x, y_y = int(y_x * increase), -int(y_y * increase)
    # print (px, py)

    print (image.shape)
    cv2.imwrite('test.png', image)
    image = cv2.imread('test.png')
    print (image.shape)

    cv2.circle(image, (int(cx), int(cy)), 5, (0, 0, 255), -1)
    cv2.line(image, (int(cx), int(cy)), (int(cx + y_x), int(cy + y_y)), (255, 0, 0), 3)

    cv2.imshow("eye", image)
    # cv2.imshow("right_eye", right_eye)
    # cv2.imshow("left_eye", left_eye)
    cv2.waitKey(0)
    # raise "debug"

def main():
    logger.info(json.dumps(vars(args), indent=2))
    save_to_record(str(json.dumps(vars(args))) + "\n")

    # TensorBoard SummaryWriter
    writer = SummaryWriter() if args.tensorboard else None

    # set random seed
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # create output directory
    outdir = args.outdir
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    outpath = os.path.join(outdir, 'config.json')
    with open(outpath, 'w') as fout:
        json.dump(vars(args), fout, indent=2)

    # data loaders
    train_loader, test_loader = get_loader(
        args.dataset, args.test_id, args.batch_size, args.num_workers, True)



    # for step, (images, poses, gazes) in enumerate(train_loader):
    #     print ("images.shape: ", images.shape)
    #     for index in range(len(images)):
    #
    #         image = np.asarray(images[index]).astype(np.uint8).copy()
    #
    #         # image = np.ascontiguousarray(image, dtype = np.uint8)
    #         # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #         gaze = np.asarray(gazes[index])
    #         pose = np.asarray(poses[index])
    #
    #         image = image.transpose(1, 2, 0)
    #
    #         height, width, channels = image.shape
    #         cy, cx = height/2, width/2
    #
    #         print image.shape
    #         print type(image)
    #
    #         print "gaze: ", gaze
    #         print "pose: ", pose
    #
    #         # cv2.imshow("image", image)
    #         # cv2.waitKey(0)
    #
    #         plot_gaze_pose([cx, cy], gaze, pose, image)
    #
    #     raise ("debug")

    # model
    module = importlib.import_module('models.{}'.format(args.arch))
    model = module.Model()

    # weights = "models/resnet10_weights.npy"
    # print "loading: ", weights
    # model = module.Model(weights)

    model.cuda()

    criterion = nn.MSELoss(size_average=True)
    # criterion = nn.SmoothL1Loss(size_average=True)

    # optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.base_lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=args.nesterov)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(
    #     optimizer, milestones=args.milestones, gamma=args.lr_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=args.lr_decay, last_epoch=-1)

    config = {
        'tensorboard': args.tensorboard,
        'tensorboard_images': args.tensorboard_images,
        'tensorboard_parameters': args.tensorboard_parameters,
    }

    # args.pretrained_path = "results/2018-12-05-10-30_test_data_resnet34_pretrained_0.001_relu_l2_4_camera_batch10/model_state.pth"
    #
    # if args.pretrained_path != "":
    #     state_dict = torch.load(args.pretrained_path)
    #     model.load_state_dict(state_dict)
    #
    #     print ("args.pretrained_path: ", args.pretrained_path)
    #     # raise ("debug")

    # run test before start training
    test(0, model, criterion, test_loader, config, writer)

    for epoch in range(1, args.epochs + 1):
        scheduler.step()

        lr = scheduler.get_lr()
        print("current learnin rate: ", str(lr))
        save_to_record("current learnin rate: " + str(lr))

        train(epoch, model, optimizer, criterion, train_loader, config, writer)
        angle_error = test(epoch, model, criterion, test_loader, config,
                           writer)

        state = OrderedDict([
            ('args', vars(args)),
            ('state_dict', model.state_dict()),
            ('optimizer', optimizer.state_dict()),
            ('epoch', epoch),
            ('angle_error', angle_error),
        ])

        if epoch % 50 == 0:
            model_path = os.path.join(outdir, 'model_state_' + str(epoch) + '.pth')
            # torch.save(model.state_dict(), model_path)
            torch.save(state, model_path)

    if args.tensorboard:
        outpath = os.path.join(outdir, 'all_scalars.json')
        writer.export_scalars_to_json(outpath)


if __name__ == '__main__':
    main()
