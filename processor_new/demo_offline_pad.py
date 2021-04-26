#!/usr/bin/env python
import os
import sys
import argparse
import json
import shutil
import time
import pdb
import numpy as np
import torch
import skvideo.io

from .io import IO
import tools
import tools.utils as utils

import cv2

max_frame = 60

class DemoOffline(IO):

    def start(self):
        
        # initiate
        label_name_path = './config/st_gcn/guard/V7/xviewnobias/label.txt'
        with open(label_name_path) as f:
            label_name = f.readlines()
            label_name = [line.rstrip() for line in label_name]
            self.label_name = label_name

        # pose estimation
        video, data_numpy = self.pose_estimation()
        # 此处data_numpy的shape为(3,#frames_len,18,1)
        # pdb.set_trace()
        # action recognition
        data = torch.from_numpy(data_numpy)
        data = data.unsqueeze(0)
        data = data.float().to(self.dev).detach()  # (1, channel, frame, joint, person)

        # model predict
        # 到这之前还是全部帧的信息，说明问题出在predict
        voting_label_name, video_label_name, output, intensity = self.predict(data)
        # pdb.set_trace()
        # 显示change
        # render the video
        images = self.render_video(data_numpy, voting_label_name,
                            video_label_name, intensity, video)
        # images = self.render_video(data_numpy, video_label_name,
        #                     voting_label_name, intensity, video)

        # visualize
        for image in images:
            image = image.astype(np.uint8)
            cv2.imshow("ST-GCN", image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    # def predict(self, data):
    #     # forward
    #     # pdb.set_trace()
    #     output, feature = self.model.extract_feature(data)
    #     output = output[0]
    #     feature = feature[0]
    #     intensity = (feature*feature).sum(dim=0)**0.5
    #     intensity = intensity.cpu().detach().numpy()

    #     # get result
    #     # classification result of the full sequence
    #     voting_label = output.sum(dim=3).sum(
    #         dim=2).sum(dim=1).argmax(dim=0)
    #     voting_label_name = self.label_name[voting_label]
    #     # classification result for each person of the latest frame
    #     num_person = data.size(4)
    #     latest_frame_label = [output[:, :, :, m].sum(
    #         dim=2)[:, -1].argmax(dim=0) for m in range(num_person)]
    #     latest_frame_label_name = [self.label_name[l]
    #                                for l in latest_frame_label]

    #     num_person = output.size(3)
    #     num_frame = output.size(1)
    #     video_label_name = list()
    #     for t in range(num_frame):
    #         frame_label_name = list()
    #         for m in range(num_person):
    #             # t是某一帧 person_label是固定了t和m的结果 t为一帧 m为某人 所以多帧拼接就改这边就好了
    #             person_label = output[:, t, :, m].sum(dim=1).argmax(dim=0)
    #             person_label_name = self.label_name[person_label]
    #             frame_label_name.append(person_label_name)
    #         video_label_name.append(frame_label_name)
    #     return voting_label_name, video_label_name, output, intensity

    def predict(self, data):
        # forward
        # pdb.set_trace()
        # 写死
        # add
        # data_in = torch.zeros(1,3,300,18,2)
        # data_in[:,:,0:data.shape[2],:,0:data.shape[4]] = data
        # data_in = data_in.float().to(self.dev).detach()
        # add_finished
        # output, feature = self.model.extract_feature(data)
        # change
        # pdb.set_trace()
        _temp = self.model(data)
        print(_temp)
        output, feature = self.model.extract_feature(data)
        output = output[0]
        feature = feature[0]
        intensity = (feature*feature).sum(dim=0)**0.5
        intensity = intensity.cpu().detach().numpy()

        # get result
        # classification result of the full sequence

        print(output.sum(dim=3).sum(dim=2).sum(dim=1))


        voting_label = output.sum(dim=3).sum(
            dim=2).sum(dim=1).argmax(dim=0)
        voting_label_name = self.label_name[voting_label]
        # classification result for each person of the latest frame
        # change
        num_person = data.size(4)
        # num_person = data_in.size(4)
        latest_frame_label = [output[:, :, :, m].sum(
            dim=2)[:, -1].argmax(dim=0) for m in range(num_person)]
        latest_frame_label_name = [self.label_name[l]
                                   for l in latest_frame_label]

        num_person = output.size(3)
        num_frame = output.size(1)
        video_label_name = list()
        for t in range(num_frame):
            frame_label_name = list()
            if t == 0:
                for m in range(num_person):
                # t是某一帧 person_label是固定了t和m的结果 t为一帧 m为某人 所以多帧拼接就改这边就好了
                    person_label = output[:, t, :, m].sum(dim=1).argmax(dim=0)
                    person_label_name = self.label_name[person_label]
                    frame_label_name.append(person_label_name)
                video_label_name.append(frame_label_name)
            if t!=0 and t%30==0:
                for m in range(num_person):
                    # t是某一帧 person_label是固定了t和m的结果 t为一帧 m为某人 所以多帧拼接就改这边就好了
                    person_label = output[:, t-29:t+1, :, m].sum(dim=2).sum(dim=1).argmax(dim=0)
                    person_label_name = self.label_name[person_label]
                    frame_label_name.append(person_label_name)
                video_label_name.append(frame_label_name)
            else:
                video_label_name.append(video_label_name[-1])
        return voting_label_name, video_label_name, output, intensity

    def render_video(self, data_numpy, voting_label_name, video_label_name, intensity, video):
        images = utils.visualization.stgcn_visualize(
            data_numpy,
            self.model.graph.edge,
            intensity, video,
            voting_label_name,
            video_label_name,
            self.arg.height)
        return images

    def pose_estimation(self):
        # load openpose python api
        if self.arg.openpose is not None:
            sys.path.append('{}/python'.format(self.arg.openpose))
            sys.path.append('{}/build/python'.format(self.arg.openpose))
        try:
            # 加了一行
            sys.path.append('/usr/local/python')
            from openpose import pyopenpose as op
        except:
            print('Can not find Openpose Python API.')
            return


        video_name = self.arg.video.split('/')[-1].split('.')[0]

        # initiate
        opWrapper = op.WrapperPython()
        params = dict(model_folder='./models', model_pose='COCO')
        opWrapper.configure(params)
        opWrapper.start()
        self.model.eval()
        video_capture = cv2.VideoCapture(self.arg.video)
        video_length = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        pose_tracker = naive_pose_tracker(data_frame=video_length)

        # pose estimation
        start_time = time.time()
        frame_index = 0
        video = list()

        # iii = 0
        while(True):

            # get image
            ret, orig_image = video_capture.read()
            if orig_image is None:
                break
            # change
            # save_path = "./A40Pic/{:06d}.jpg".format(iii)
            # cv2.imwrite(save_path, orig_image)
            # iii+=1
            # origin code 

            source_H, source_W, _ = orig_image.shape
            orig_image = cv2.resize(
                orig_image, (256 * source_W // source_H, 256))
            H, W, _ = orig_image.shape
            video.append(orig_image)

            # origin code

            # # change code
            # H, W, _ = orig_image.shape
            # video.append(orig_image)
            # # change code 不进行resize 直接估计点(同训练集一样的处理方式)

            # pose estimation
            datum = op.Datum()
            datum.cvInputData = orig_image
            # datum.cvInputData = cv2.imread(save_path)
            # 原始
            opWrapper.emplaceAndPop([datum])
            # opWrapper.emplaceAndPop(op.VectorDatum[datum])
            multi_pose = datum.poseKeypoints  # (num_person, num_joint, 3)
            # pdb.set_trace()
            if len(multi_pose.shape) != 3:
                continue

            # normalization
            multi_pose[:, :, 0] = multi_pose[:, :, 0]/W
            multi_pose[:, :, 1] = multi_pose[:, :, 1]/H
            multi_pose[:, :, 0:2] = multi_pose[:, :, 0:2] - 0.5
            # multi_pose[:, :, 0:2] = multi_pose[:, :, 0:2]
            multi_pose[:, :, 0][multi_pose[:, :, 2] == 0] = 0
            multi_pose[:, :, 1][multi_pose[:, :, 2] == 0] = 0

            # pose tracking
            pose_tracker.update(multi_pose, frame_index)
            frame_index += 1

            print('Pose estimation ({}/{}).'.format(frame_index, video_length))
        # pdb.set_trace()
        data_numpy = pose_tracker.get_skeleton_sequence()
        return video, data_numpy

    @staticmethod
    def get_parser(add_help=False):

        # parameter priority: command line > config > default
        parent_parser = IO.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Demo for Spatial Temporal Graph Convolution Network')

        # region arguments yapf: disable
        parser.add_argument('--video',
                            default='./resource/media/skateboarding.mp4',
                            help='Path to video')
        parser.add_argument('--openpose',
                            default=None,
                            help='Path to openpose')
        parser.add_argument('--model_input_frame',
                            default=128,
                            type=int)
        parser.add_argument('--model_fps',
                            default=30,
                            type=int)
        parser.add_argument('--height',
                            default=1080,
                            type=int,
                            help='height of frame in the output video.')
        parser.set_defaults(
            config='./config/st_gcn/guard/V7/xsub/demo_offline.yaml')
        parser.set_defaults(print_log=False)
        # endregion yapf: enable

        return parser

class naive_pose_tracker():
    """ A simple tracker for recording person poses and generating skeleton sequences.
    For actual occasion, I recommend you to implement a robuster tracker.
    Pull-requests are welcomed.
    """

    def __init__(self, data_frame=128, num_joint=18, max_frame_dis=np.inf):
        self.data_frame = data_frame
        self.num_joint = num_joint
        self.max_frame_dis = max_frame_dis
        # self.latest_frame = 0
        self.latest_frame = -1
        self.trace_info = list()

    def update(self, multi_pose, current_frame):
        # multi_pose.shape: (num_person, num_joint, 3)

        if current_frame <= self.latest_frame:
            return

        if len(multi_pose.shape) != 3:
            return

        score_order = (-multi_pose[:, :, 2].sum(axis=1)).argsort(axis=0)
        for p in multi_pose[score_order]:

            # match existing traces
            matching_trace = None
            matching_dis = None
            for trace_index, (trace, latest_frame) in enumerate(self.trace_info):
                # trace.shape: (num_frame, num_joint, 3)
                if current_frame <= latest_frame:
                    continue
                mean_dis, is_close = self.get_dis(trace, p)
                if is_close:
                    if matching_trace is None:
                        matching_trace = trace_index
                        matching_dis = mean_dis
                    elif matching_dis > mean_dis:
                        matching_trace = trace_index
                        matching_dis = mean_dis

            # update trace information
            if matching_trace is not None:
                trace, latest_frame = self.trace_info[matching_trace]

                # padding zero if the trace is fractured
                pad_mode = 'interp' if latest_frame == self.latest_frame else 'zero'
                pad = current_frame-latest_frame-1
                new_trace = self.cat_pose(trace, p, pad, pad_mode)
                self.trace_info[matching_trace] = (new_trace, current_frame)

            else:
                new_trace = np.array([p])
                self.trace_info.append((new_trace, current_frame))

        self.latest_frame = current_frame

    def get_skeleton_sequence(self):

        # remove old traces
        valid_trace_index = []
        for trace_index, (trace, latest_frame) in enumerate(self.trace_info):
            if self.latest_frame - latest_frame < self.data_frame:
                valid_trace_index.append(trace_index)
        self.trace_info = [self.trace_info[v] for v in valid_trace_index]

        num_trace = len(self.trace_info)
        if num_trace == 0:
            return None

        data = np.zeros((3, max_frame, self.num_joint, num_trace))
        for trace_index, (trace, latest_frame) in enumerate(self.trace_info):
            end = self.data_frame - (self.latest_frame - latest_frame)
            # pdb.set_trace()
            if end > max_frame:
                bg = (end - max_frame)//2
                ed = bg + max_frame
                d = trace[bg:ed]
            else:
                d = trace[-end:]
            # beg = end - len(d)
            beg = 0
            # pdb.set_trace()
            # print('-'*100)
            # print('{}:{}'.format(beg,end))
            data[:, beg:end, :, trace_index] = d.transpose((2, 0, 1))
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # res = np.zeros((data.shape[0], 60, data.shape[2], data.shape[3]))
        # res[:,0:data.shape[1],:,:] = data
        # return res
        pdb.set_trace()
        return data

    # concatenate pose to a trace
    def cat_pose(self, trace, pose, pad, pad_mode):
        # trace.shape: (num_frame, num_joint, 3)
        num_joint = pose.shape[0]
        num_channel = pose.shape[1]
        if pad != 0:
            if pad_mode == 'zero':
                trace = np.concatenate(
                    (trace, np.zeros((pad, num_joint, 3))), 0)
            elif pad_mode == 'interp':
                last_pose = trace[-1]
                coeff = [(p+1)/(pad+1) for p in range(pad)]
                interp_pose = [(1-c)*last_pose + c*pose for c in coeff]
                trace = np.concatenate((trace, interp_pose), 0)
        new_trace = np.concatenate((trace, [pose]), 0)
        return new_trace

    # calculate the distance between a existing trace and the input pose

    def get_dis(self, trace, pose):
        last_pose_xy = trace[-1, :, 0:2]
        curr_pose_xy = pose[:, 0:2]

        mean_dis = ((((last_pose_xy - curr_pose_xy)**2).sum(1))**0.5).mean()
        wh = last_pose_xy.max(0) - last_pose_xy.min(0)
        scale = (wh[0] * wh[1]) ** 0.5 + 0.0001
        is_close = mean_dis < scale * self.max_frame_dis
        return mean_dis, is_close
