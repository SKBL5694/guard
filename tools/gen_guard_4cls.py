import os
import sys
import pickle
import pdb
import argparse
import numpy as np
from numpy.lib.format import open_memmap

from utils.ntu_read_skeleton import read_xyz
from utils.ntu_read_skeleton_new import read_xyc, read_xycc,read_xyccc

training_subjects = [
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38
]
training_cameras = [2, 3]
max_body = 1
num_joint = 18
max_frame = 60
toolbar_width = 30

def print_toolbar(rate, annotation=''):
    # setup toolbar
    sys.stdout.write("{}[".format(annotation))
    for i in range(toolbar_width):
        if i * 1.0 / toolbar_width > rate:
            sys.stdout.write(' ')
        else:
            sys.stdout.write('-')
        sys.stdout.flush()
    sys.stdout.write(']\r')


def end_toolbar():
    sys.stdout.write("\n")


def gendata(data_path,
            out_path,
            ignored_sample_path=None,
            benchmark='xview',
            part='eval',
            chosen = None,
            max_frame = None):
    if ignored_sample_path != None:
        with open(ignored_sample_path, 'r') as f:
            ignored_samples = [
                line.strip() + '.skeleton' for line in f.readlines()
            ]
    else:
        ignored_samples = []
    sample_name = []
    sample_label = []
    for filename in os.listdir(data_path):
        if filename in ignored_samples:
            continue
        action_class = int(
            filename[filename.find('A') + 1:filename.find('A') + 4])
        subject_id = int(
            filename[filename.find('P') + 1:filename.find('P') + 4])
        camera_id = int(
            filename[filename.find('C') + 1:filename.find('C') + 4])
        action2label = {31:0,36:1,38:2,40:3}
        if benchmark == 'xview':
            istraining = (camera_id in training_cameras)
        elif benchmark == 'xsub':
            istraining = (subject_id in training_subjects)
        else:
            raise ValueError()

        if part == 'train':
            issample = istraining
        elif part == 'val':
            issample = not (istraining)
        else:
            raise ValueError()

        if issample:
            sample_name.append(filename)
            sample_label.append(action2label[action_class])
    # pdb.set_trace()
    with open('{}/{}_label.pkl'.format(out_path, part), 'wb') as f:
        pickle.dump((sample_name, list(sample_label)), f)
    # np.save('{}/{}_label.npy'.format(out_path, part), sample_label)

    fp = open_memmap(
        '{}/{}_data.npy'.format(out_path, part),
        dtype='float32',
        mode='w+',
        shape=(len(sample_label), 3, max_frame, num_joint, max_body))
    # fp(样本数, 3, 最大帧(300), 骨骼数, 个体数)
    for i, s in enumerate(sample_name):
        print_toolbar(i * 1.0 / len(sample_label),
                      '({:>5}/{:<5}) Processing {:>5}-{:<5} data: '.format(
                          i + 1, len(sample_name), benchmark, part))
        # data(3, num_frame, num_joint, max_body)
        data = read_xyccc(
            os.path.join(data_path, s), max_body=max_body, num_joint=num_joint, chosen=chosen, max_frame=max_frame) # !!!!!max_body在read_xyc中被改了
        # 把读取到的数据放在 第i个样本的全部channel，'前样本帧数'层(骨骼和个体自然也全选)
        fp[i, :, 0:data.shape[1], :, :] = data
    end_toolbar()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='NTU-RGB-D Data Converter.')
    parser.add_argument(
        '--data_path', default='data/NTU-RGB-D/nturgb+d_skeletons')
    # parser.add_argument(
    #     '--ignored_sample_path',
    #     default='resource/NTU-RGB-D/samples_with_missing_skeletons.txt')
    parser.add_argument('--out_folder', default='data/NTU-RGB-D')

    benchmark = ['xsub', 'xview']
    part = ['train', 'val']
    arg = parser.parse_args()

    a107 = os.listdir('/home/zy/data/guard/filter_1person_from_2/A107_over')
    a110 = os.listdir('/home/zy/data/guard/filter_1person_from_2/A110_over')
    templist0 = a107 + a110
    templist1 = [x[:-8] for x in templist0]
    chosen_num = [int(x[-5]) for x in templist0]
    chosendict = {key:val for key,val in zip(templist1,chosen_num)}
    for b in benchmark:
        for p in part:
            out_path = os.path.join(arg.out_folder, b)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            gendata(
                arg.data_path,
                out_path,
                None,
                benchmark=b,
                part=p,
                chosen=chosendict,
                max_frame=max_frame)