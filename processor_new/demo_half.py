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

class DemoRealtime(IO):
    """ A demo for utilizing st-gcn in the realtime action recognition.
    The Openpose python-api is required for this demo.

    Since the pre-trained model is trained on videos with 30fps,
    and Openpose is hard to achieve this high speed in the single GPU,
    if you want to predict actions by **camera** in realtime,
    either data interpolation or new pre-trained model
    is required.

    Pull requests are always welcome.
    """

    def start(self, write_jpg = False):
        # load openpose python api
        if self.arg.openpose is not None:
            sys.path.append('{}/python'.format(self.arg.openpose))
            sys.path.append('{}/build/python'.format(self.arg.openpose))
        try:
            sys.path.append('/usr/local/python')
            from openpose import pyopenpose as op
        except:
            print('Can not find Openpose Python API.')
            return

        video_name = self.arg.video.split('/')[-1].split('.')[0]
        label_name_path = './config/st_gcn/guard/V7/xviewnobias/label.txt'
        # label_name_path = './resource/kinetics_skeleton/label_name.txt'
        with open(label_name_path) as f:
            label_name = f.readlines()
            label_name = [line.rstrip() for line in label_name]
            self.label_name = label_name

        # initiate
        opWrapper = op.WrapperPython()
        params = dict(model_folder='./models', model_pose='COCO')
        # params = dict(model_folder='./models', model_pose='BODY_25')
        opWrapper.configure(params) 
        opWrapper.start()
        self.model.eval()
        pose_tracker = naive_pose_tracker()
        # pdb.set_trace()
        if self.arg.video == 'camera_source':
            video_capture = cv2.VideoCapture(0)
            # video_capture.set(5,60)
        else:
            video_capture = cv2.VideoCapture(self.arg.video)

        # start recognition
        start_time = time.time()
        frame_index = 0
        while(True):
            # print('------------------------------------------------------')
            # print(video_capture.get(5))
            # print('------------------------------------------------------')
            tic = time.time()

            # get image
            ret, orig_image = video_capture.read()
            if orig_image is None:
                break
            source_H, source_W, _ = orig_image.shape
            orig_image = cv2.resize(
                orig_image, (256 * source_W // source_H, 256))
            H, W, _ = orig_image.shape
            
            # pose estimation
            datum = op.Datum()
            datum.cvInputData = orig_image
            opWrapper.emplaceAndPop([datum])
            multi_pose = datum.poseKeypoints  # (num_person, num_joint, 3) 3应该是x,y和置信度 坐标原点在图像左上角 
            # print('aaa')
            if len(multi_pose.shape) != 3:
                # print('bbb')
                continue

            # normalization 坐标均值归一化
            multi_pose[:, :, 0] = multi_pose[:, :, 0]/W   # 0是X坐标 所以除以W
            multi_pose[:, :, 1] = multi_pose[:, :, 1]/H
            multi_pose[:, :, 0:2] = multi_pose[:, :, 0:2] - 0.5 # 减去均值 x，y已经缩放到[0,1]范围，但这个均值是不是可以通过计算得出呢，而非硬编码0.5？ 不包含置信度
            #首先置信度为零的点坐标默认为(0,0)，可能就是18个预设点的某些点检测不到，就设置为(0,0),前面减均值的时候，将这类坐标都变成了-0.5,以下两步是将这些置信度为0的坐标改回(0,0)
            multi_pose[:, :, 0][multi_pose[:, :, 2] == 0] = 0  
            multi_pose[:, :, 1][multi_pose[:, :, 2] == 0] = 0
            # print('aaa')
            # pose tracking
            if self.arg.video == 'camera_source':
                # 改
                # frame_index = int((time.time() - start_time)*self.arg.fps)
                frame_index = int((time.time() - start_time)*10)
                # frame_index += 1
            else:
                frame_index += 1
            pose_tracker.update(multi_pose, frame_index)
            # 貌似这种更新方法很容易跟踪错人(感觉上是，因为是用历史trace_info的骨骼数据中，最后一次更新的数据平均距离来对比)
            # 也就是说，用一个角色最后一次被捕获到的骨骼位置信息作为下次定位该角色的'中心'，以此来计算平均距离
            # .update之后的结果就是 self.trace_info被更新了。这个list的个数为出现过的角色数，与update前相比，被更新的就是还在视野中的角色当前帧的信息，更新方式为，将其信息的shape(num_index-1,18,3)也就是前一帧结果 变为(num_index,18,3)
            # 如果某个角色中间信息缺失，再回来的话，则进行补帧，总之只要被update了，骨骼数据的第一维度就得是帧数
            # if frame_index == 63:
                # pdb.set_trace()
            data_numpy = pose_tracker.get_skeleton_sequence()
            # (3,data_frame=128,18,trace_num)
            data = torch.from_numpy(data_numpy)
            data = data.unsqueeze(0)
            data = data.float().to(self.dev).detach()  # (1, channel, frame, joint, person)
            # pdb.set_trace()
            # model predict
            voting_label_name, video_label_name, output, intensity = self.predict(
                data)
            # pdb.set_trace()
            # visualization
            app_fps = 1 / (time.time() - tic)
            image = self.render(data_numpy, voting_label_name,
                                video_label_name, intensity, orig_image, app_fps)
            if write_jpg:
                cv2.imwrite('./demo/{}.jpg'.format(frame_index), image)
            cv2.imshow("ST-GCN", image)
            # print(time.time()-tic)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def predict(self, data):
        # forward
        # pdb.set_trace()
        output, feature = self.model.extract_feature(data)
        output = output[0]  #取batch中第一个样本
        feature = feature[0]
        intensity = (feature*feature).sum(dim=0)**0.5    #相当于feature的每个元素平方把256个channel加在一块再开根号  估计是用来搞gradcam的
        intensity = intensity.cpu().detach().numpy()

        # get result
        # classification result of the full sequence
        voting_label = output.sum(dim=3).sum(
            dim=2).sum(dim=1).argmax(dim=0)
        voting_label_name = self.label_name[voting_label]
        # classification result for each person of the latest frame
        num_person = data.size(4)
        latest_frame_label = [output[:, :, :, m].sum(
            dim=2)[:, -1].argmax(dim=0) for m in range(num_person)]  #每个人最后一帧的类别最大值
        # pdb.set_trace()
        latest_frame_label_name = [self.label_name[l]
                                   for l in latest_frame_label]

        num_person = output.size(3)
        num_frame = output.size(1)
        video_label_name = list()
        for t in range(num_frame):
            frame_label_name = list()
            for m in range(num_person):
                person_label = output[:, t, :, m].sum(dim=1).argmax(dim=0)
                person_label_name = self.label_name[person_label]
                frame_label_name.append(person_label_name)
            video_label_name.append(frame_label_name)   # video_label_name长度是所谓的num_frame(32)，也不知道哪来的 每个元素是某一帧中的 每个人的标签 某个元素eg: 3个人 [0,5,1]
        return voting_label_name, video_label_name, output, intensity

    def render(self, data_numpy, voting_label_name, video_label_name, intensity, orig_image, fps=0):
        images = utils.visualization.stgcn_visualize(
            data_numpy[:, [-1]],
            self.model.graph.edge,
            intensity[[-1]], [orig_image],
            voting_label_name,
            [video_label_name[-1]],
            self.arg.height,
            fps=fps)
        image = next(images)
        image = image.astype(np.uint8)
        return image

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
                            # default=30,
                            default=30,
                            type=int)
        parser.add_argument('--height',
                            default=1080,
                            type=int,
                            help='height of frame in the output video.')
        # checkpoint kinetics
        parser.set_defaults(
            config='./config/st_gcn/guard/V7/xsub/demo_realtime.yaml')
            # config='./config/st_gcn/kin_new/demo_realtime.yaml')
        parser.set_defaults(print_log=False)
        # endregion yapf: enable

        return parser

class naive_pose_tracker():
    """ A simple tracker for recording person poses and generating skeleton sequences.
    For actual occasion, I recommend you to implement a robuster tracker.
    Pull-requests are welcomed.
    """

    def __init__(self, data_frame=60, num_joint=18, max_frame_dis=np.inf):
    # def __init__(self, data_frame=128, num_joint=25, max_frame_dis=np.inf):
        self.data_frame = data_frame
        self.num_joint = num_joint
        self.max_frame_dis = max_frame_dis
        self.latest_frame = 0
        self.trace_info = list()

    def update(self, multi_pose, current_frame):
        # multi_pose.shape: (num_person, num_joint, 3)

        if current_frame <= self.latest_frame:
            return

        if len(multi_pose.shape) != 3:
            return
        # pdb.set_trace()
        # 将18个点的置信度加在一起并按顺序排列，负号是为了将置信度大的排在前面，也就是下面的p(trace_info)中，置信度越高的越靠前
        score_order = (-multi_pose[:, :, 2].sum(axis=1)).argsort(axis=0) 
        # p的shape是二维(18,3)


        """
        总体更新思路如下：
        第一帧检测到n个人，则info中就会有n个位置
        从第二帧开始,1)假设pose给了小于等于n个人，也就是p循环小于n次;每次拿过来一个p的信息，和info中的信息逐个对比，找到info中，离p最近的
        那个信息所对应的位置，就认为当前p和往该位置表达的是同一个人;在该位置中加入p的信息，并将该位置的latest_frame更改为current_frame(latest_frame可理解为该位置所代表的人物最新一次出现的帧号)
        如此一来，下一轮循环的p所对比的位置就少了一个;也就是if currnet_frame <= latest_frame; continue所起的作用(因为在前一个p时,latest_frame已经被更新成current_frame相等的值了)
        一言以蔽之：本轮置信度高的(p)先找位置，且一定能找到位置，无论合理与否，之后这个位置在这一帧就不会再被拿出来了。所以is_close就起到了非常关键的作用，设置好的话可以一定程度防止该事情的发生
        但目前来看没起作用,因为is_close恒为true。可通过更改初始化方法中的max_frame_dis的值来实现
                   2)假设pose给了大于n个人，从上面的情况可以明显的看出，置信度最小的p会被当作新的人加入info中，哪怕在之前的帧中，与其极其邻近
        的位置有人出现过(我的意思就是本来就是同一个人，在自己的地方基本没动，但因为置信度排名过低，在n名之后，则这个人在当前帧就会被当成个新的人，
        占据一个新的位置，接踵而来两个问题：其一是他自己的轨迹断掉了；其二是：如果新来的某个或者某几个人置信度过高，会导致所有的轨迹全部混乱，
        因为置信度高的先选，如果没有close判定，则可能导致上述情况。
                   3)最后就是我之前想到的问题ABCD四个人，开始只有ABC，后来C走了，再后来D进来了。情况好的话(其实也很坏)D会被和之前C的位置
        连起来；更坏的情况是D的置信度很高，他先选，把AB中的位置给抢了，最后都乱了。
        最后就是两点 1：置信度到底怎么确定的(openpose给的)，如果这个运气好的话，不发生置信度颠倒的事情，那情况会好一些 
                   2：搞不定置信度的话，close判定就非常重要了  

        最终data的输出维度为(3,data_frame,18,num_trace)
        其中data_frame在offline中就是视频长度，在实时检测中自己定义(default=128)，num_trace为128帧以内最大同屏人数(完美版应该是出现过的不同的人数,但因为confidence和is_close尚未高清，所以没法实现)

        """




        for p in multi_pose[score_order]:

            # match existing traces
            matching_trace = None
            matching_dis = None
            for trace_index, (trace, latest_frame) in enumerate(self.trace_info): 
                # 这句话貌似随着人增多，执行次数增多，1+2+3+4+5+………… 虽然一直continue，但好像也挺蠢。 to do
                # trace.shape: (num_frame, num_joint, 3)
                # 这句话使得self.trace_info.append能够多次执行，因为这句话下面的continue跳过了给matching_trace赋值的部分
                # 首先latest_frame和self.latest_frame是两个变量 ;正常来讲，current 会比 self.latest大1(不计算时间)
                # 只有当前帧在识别到多个人时,即p有2个及以上循环时,上面的for trace_index循环才有第二次触发的机会，latest_frame值被改变，下面的这个if才会为true
                # 因为第一帧，第一轮self.trace_info信息是空的。latest_frame为None，全为空的情况会把检测到的第一个人的信息直接加到self.trace_info中，此时如果没有第二个人，则本帧也就处理完了 合理
                # 只有当下面这个if 为 ture时，self.trace_info才能继续加入后续新的人的信息
                # 下面这个if中 current_frame在每次for循环中是不会变化的，只有latest_frame会变化，取决于枚举trace_info时候
                # 只要枚举的是当前帧的信息，则下面if会变为true，continue后面的语句就不会触发，matching_trace就始终为None，self.trace_info就不断append，直到该帧中所有的人都被append之后再说 合理
                # 那么问题就变成self.trace_info如何更新的问题
                if current_frame <= latest_frame: 
                    continue
                # 对于continue下面这段以及 update trace information 中if下面的部分，在第一帧不会被执行到


                # 因该是对比当前帧骨骼点p和前一(n)帧已存在骨骼点(info中的数据)的距离，看起来is_close是用来判断是不是离太远的(太远直接比较都不比较了)
                # 离得近的话就保留下来，进入下面的matching_trace和比较距离环节，最终把和当前帧的这个骨骼距离最相近的trace_index保留下来，作为matching_trace
                # 说白了就是给当前帧检测出来的角色A的骨骼找到上一(n)帧的对应的角色A的骨骼
                # to do 问题在于会不会存在当前两个不同的人对到上一帧相同的人的事情（目前看是有可能的）
                mean_dis, is_close = self.get_dis(trace, p)
                # 以下讨论的部分均是从第二帧开始的
                # 这部分的is_close非常重要，本来ABC三个人，C出去了，D进来了，如果没有is_close，就不会加入第四个trace;有的话都有可能家不进去(貌似D从c出去的地方进来就加不进去)
                if is_close: # 下面这个if elif干的一样的事情   换句话说，只有在matching_trace < trace_index时才啥都不干
                    # 首先，第一个人来的时候matching_trace为none，于是matching_trace和matching_dis变成了第一个人的序号和距离
                    if matching_trace is None:
                        matching_trace = trace_index
                        matching_dis = mean_dis
                    # 当第二个人来的时候，如果他的平均距离小于第一个人的，就把matching_trace和matching_dis更改为第二个人的，否则不变化
                    # 换句话说，if is_close下的判断最终起到的目的是将matching_trace和matching_dis的值设置为mean_dis最小的那个人在self.trace_info中的index以及对应的mean_dis
                    elif matching_dis > mean_dis:
                        matching_trace = trace_index
                        matching_dis = mean_dis

            # update trace information
            # 把当前帧的一个骨骼信息和前一帧的info对比了一圈之后，将挑选出对应(距离最近)的骨骼信息和帧号拿出来给trace和latest_frame
            # 本来的trace和latest_frame是遍历完info之后最后一个角色的trace和latest_frame，我们要将其更新为我们找到的信息
            if matching_trace is not None:
                trace, latest_frame = self.trace_info[matching_trace]

                # padding zero if the trace is fractured
                # 这个时候的self.latest_frame还没更新，也就是说还是前一帧;而latest_frame是info中同当前帧最吻合的骨骼所在帧
                # 如果二者不想等，那么说明同当前骨骼最吻合的不是前一帧
                # 所以这个pad_mode的判据是 “同 当前帧的当前骨骼 平均距离最近的骨骼 是不是来自于前一帧” 是则等号成立 反之不成立
                pad_mode = 'interp' if latest_frame == self.latest_frame else 'zero'
                # pad为与当前骨骼数据最近的那帧和当前帧中间的间隔帧数：0表示紧邻，1表示中间有1帧，n表示中间有n帧
                pad = current_frame-latest_frame-1
                # 将前一帧和这一帧的pose拼接在一起
                # trace为info中与当前骨骼数据最近的骨骼数据 p为当前帧识别到的一个骨骼数据 (18,3)
                # pad为最近数据所在帧到前一帧的’帧距‘ 
                # 在每论currentframe+1的情况下：若pad为0,则pad_mode为interp 否则为zero
                # n = “p所在的帧-距离p最近数据所在的帧+1” 即从最近帧到当前帧的数目，前提为cat_pose中，trace的第一维不是注释中的num_frame，而是1
                # 换句话说就是info中的骨骼数据到底是怎么回事儿?
                # 看了看下面的info更新，info中的trace的第一维貌似是不断增长的，只要这个人还在视野中，每处理一帧，info中对应信息的第一维就+1
                # 所以说n其实就是角色在视野中的帧数(数到当前时刻)，n = info中该角色骨骼数据的帧数 + pad(间隔) + 1(当前)
                # new_trace(n,18,3)
                new_trace = self.cat_pose(trace, p, pad, pad_mode)
                self.trace_info[matching_trace] = (new_trace, current_frame)

            else:
                new_trace = np.array([p])
                # self.trace_info是列表，数据为2tuple形式。(单人(可能是单帧，后续会加帧)骨骼点坐标(1,18,3),frame_index)
                # 在第一帧的时候 self.trace_info
                self.trace_info.append((new_trace, current_frame))

        self.latest_frame = current_frame # 处理完结后将self.latest_frame设为当前帧

    def get_skeleton_sequence(self):

        # remove old traces
        valid_trace_index = []
        for trace_index, (trace, latest_frame) in enumerate(self.trace_info):
            # 在第一帧的情况下self.latest_frame - latest_frame = 0  因为self.trace_info只保存了第一帧的信息，latest_frame全为1
            # self.latest_frame在经历了update之后就等于current_frame

             # 这里就是trace_info的删除部分，如果一个角色最新出现的帧距离当前处理的帧超过了self.data_frame，则就会被删除
             # 实现删除操作的是反过来操作的：只有小于self.data_frame的帧才会被加到valid_trace_index中，再通过其下面的info赋值语句保留
             # 换句话说，所谓valid_trace 合法轨迹 就是在self.data_frame帧内出现过的
            # print('trace_index', trace_index)
            # print('self.latest_frame:', self.latest_frame)
            # print('latest_frame', latest_frame)
            if self.latest_frame - latest_frame < self.data_frame:
                valid_trace_index.append(trace_index)
        # 找到合法的轨迹，在self.trace_info只保留合法的轨迹
        self.trace_info = [self.trace_info[v] for v in valid_trace_index]
        # print('trace_info[0].shape', self.trace_info[0][0].shape[0])
        num_trace = len(self.trace_info)
        if num_trace == 0:
            return None

        data = np.zeros((3, self.data_frame, self.num_joint, num_trace))
        for trace_index, (trace, latest_frame) in enumerate(self.trace_info):
            end = self.data_frame - (self.latest_frame - latest_frame)
            # pdb.set_trace()
            # 第一帧 暂时没看出来玄机，反正-end就很小，看起来本意是截取前几个人的数据，但首先只有一个人(1,18,3)，其次-end为-128
            # 修正：不是前几个人，而是某个人出现的前几帧
            # trace(num_frame, 18, 3)
            d = trace[-end:]
            # print(d.shape)
            # trace长度大于end时,len(d) = end
            # trace长度小于end时,len(d) = len(trace)
            beg = end - len(d)
            # data形式应该是最近data_frame帧(128)帧的数据
            # data(3, data_frame, 18, trace_indx)
            # 其中3表示(x,y,confidence);data_frame的值表示data输出最近多少帧的数据,这里为128,由于data被初始化为0,
            # 所以在赋值时：最近128帧内出现过的帧数才有值，才会被修改，没有的话就为0,begin和end就是为了找到’包含数据的帧‘在data中的位置
            # 18为骨骼点数， trace_index为视野中出现过的人物的编号
            data[:, beg:end, :, trace_index] = d.transpose((2, 0, 1))
        # 最后data输出的形式为(3(x,y,confidence),人数(推测应该是一个人连续出现的帧数),18(骨骼点数),第几个人(trace_info的顺序，应该也就是置信度顺序))
        # res = np.zeros((data.shape[0], 100, data.shape[2], data.shape[3]))
        # res[:,0:data.shape[1],:,:] = data
        # return res
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
            # interp看起来像是一种插值的方式，而不是单纯的补0
            elif pad_mode == 'interp':
                last_pose = trace[-1]
                coeff = [(p+1)/(pad+1) for p in range(pad)]
                interp_pose = [(1-c)*last_pose + c*pose for c in coeff]
                trace = np.concatenate((trace, interp_pose), 0)
        new_trace = np.concatenate((trace, [pose]), 0)
        return new_trace

    # calculate the distance between a existing trace and the input pose

    def get_dis(self, trace, pose):
        # 比较距离这个-1 就是指用骨骼数据信息中，最近被更新的一帧来比较
        last_pose_xy = trace[-1, :, 0:2]
        curr_pose_xy = pose[:, 0:2]

        mean_dis = ((((last_pose_xy - curr_pose_xy)**2).sum(1))**0.5).mean() # 18个点的平均距离
        wh = last_pose_xy.max(0) - last_pose_xy.min(0)
        scale = (wh[0] * wh[1]) ** 0.5 + 0.0001
        is_close = mean_dis < scale * self.max_frame_dis  # 由于scale为正数 self.max_frame_dis为正无穷，所以is_close必为true
        # 或许可以通过调节self.max_frame_dis限制某些东西
        return mean_dis, is_close
