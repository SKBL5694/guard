import numpy as np
import os
import json
from . import temp_tools
import pdb
# 23,24两行'res[0, i, k//3, j] = joint_info[k]/1920' 'res[1, i, k//3, j] = joint_info[k+1]/1080'需要减去均值（0.5）？
def read_xyc(file_path, max_body, num_joint=18):
	filelist = os.listdir(file_path)
	num_frame = len(filelist)
	res = np.zeros((3, num_frame, num_joint, max_body))
	flag = 0
	for i in range(num_frame):
		with open(file_path + '/' + str(i) + '_keypoints.json', 'r') as f:
			data = json.load(f)
		num_body = len(data['people'])
		if num_body > max_body:
			print('more than 2 people input, set it to 2')
			print(file_path + str(i) + "frame")
			flag = 1
			num_body = max_body
		for j in range(num_body):
			joint_info = data['people'][j]['pose_keypoints_2d']
			for k in range(0,len(joint_info),3):
				res[0, i, k//3, j] = joint_info[k]/1920
				res[1, i, k//3, j] = joint_info[k+1]/1080
				res[2, i, k//3, j] = joint_info[k+2]
	# if flag:
	# 	print(file_path)
	return res


def read_xycc(file_path, max_body, num_joint=18):
	if '054' not in file_path and '107' not in file_path and '110' not in file_path:
		max_body = 1
	filelist = os.listdir(file_path)
	num_frame = len(filelist)
	res = np.zeros((3, num_frame, num_joint, max_body))
	flag = 0
	for i in range(num_frame):
		with open(file_path + '/' + str(i) + '_keypoints.json', 'r') as f:
			data = json.load(f)
		num_body = len(data['people'])
		if num_body > max_body:
			print('more than {body} people input, set it to {body}'.format(body=max_body))
			print(file_path + str(i) + "frame")
			flag = 1
			num_body = max_body
		for j in range(num_body):
			joint_info = data['people'][j]['pose_keypoints_2d']
			for k in range(0,len(joint_info),3):
				res[0, i, k//3, j] = round(joint_info[k]/1920,3)
				res[1, i, k//3, j] = round(joint_info[k+1]/1080,3)
				res[2, i, k//3, j] = round(joint_info[k+2],3)
	res[0:2] = res[0:2] - 0.5
	res[0][res[2] == 0] = 0
	res[1][res[2] == 0] = 0

	res = temp_tools.auto_pading(res, 300) # 硬编码300, 因为max_frame没有作为参数传入

	sort_index = (-res[2, :, :, :].sum(axis=1)).argsort(axis=1)
	for t, s in enumerate(sort_index):
	    res[:, t, :, :] = res[:, t, :, s].transpose((1, 2, 0))
	res = res[:, :, :, 0:2] # 硬编码2, 输出的人数最多是2
	# if flag:
	# 	print(file_path)
	return res





















def read_xyccc(file_path, max_body, num_joint=18, chosen=None, skip=1, max_frame=None, pure_test=False, pad_strategy='zero'):
	filelist = os.listdir(file_path)
	num_frame = len(filelist)
	res = np.zeros((3, num_frame, num_joint, max_body))
	flag = 0
	# pdb.set_trace()
	if 'A107' not in file_path and 'A110' not in file_path:
		for i in range(0,num_frame,skip):
			with open(file_path + '/' + str(i) + '_keypoints.json', 'r') as f:
				data = json.load(f)
			for j in range(max_body):
				try:
					joint_info = data['people'][j]['pose_keypoints_2d']
					for k in range(0,len(joint_info),3):
						res[0, i, k//3, j] = round(joint_info[k]/1920,3)
						res[1, i, k//3, j] = round(joint_info[k+1]/1080,3)
						res[2, i, k//3, j] = round(joint_info[k+2],3)
				except:
					print(file_path)
					print(i)
					break

	else:
		pdb.set_trace() # 理应无法触发
		for i in range(0,num_frame,skip):
			with open(file_path + '/' + str(i) + '_keypoints.json', 'r') as f:
				data = json.load(f)
			name = os.path.basename(file_path)
			if name.strip() in chosen:
				j = chosen[name.strip()]
			else:
				print(name.strip())
				j = 0 
			try:
				joint_info = data['people'][j]['pose_keypoints_2d']
				for k in range(0,len(joint_info),3):
					res[0, i, k//3, 0] = round(joint_info[k]/1920,3)
					res[1, i, k//3, 0] = round(joint_info[k+1]/1080,3)
					res[2, i, k//3, 0] = round(joint_info[k+2],3)
			except:
				print(file_path)
				print(i)
				continue
	# pdb.set_trace()
	res[0:2] = res[0:2] - 0.5
	res[0][res[2] == 0] = 0
	res[1][res[2] == 0] = 0

	# res = temp_tools.auto_pading(res, 300) # 硬编码300, 因为max_frame没有作为参数传入

	sort_index = (-res[2, :, :, :].sum(axis=1)).argsort(axis=1)
	for t, s in enumerate(sort_index):
	    res[:, t, :, :] = res[:, t, :, s].transpose((1, 2, 0))
	res = res[:, :, :, 0:max_body] # 硬编码2, 输出的人数最多是2
	# if flag:
	# 	print(file_path)
	pdb.set_trace()
	ret = np.zeros((res.shape[0], max_frame, res.shape[2], res.shape[3]))
	if res.shape[1] > max_frame:
		beg = (res.shape[1]-max_frame)//2
		ret = res[:,beg:beg+max_frame,:,:]
	elif res.shape[1] == max_frame:
		ret = res
	elif res.shape[1] < max_frame:
		if pad_strategy == 'repeat':
			loop = max_frame//res.shape[1]
			for i in range(loop):
				ret[:,i*res.shape[1]:(i+1)*res.shape[1],:,:] = res
			# pdb.set_trace()
			ret[:,loop*res.shape[1]:,:,:] = res[:,:max_frame%res.shape[1],:,:]
		elif pad_strategy == 'last':
			pdb.set_trace() # 理应无法触发
			ret[:,:res.shape[1],:,:] = res
			ret[:,res.shape[1]:,:,:] = ret[:,[res.shape[1]-1],:,:].repeat(max_frame-res.shape[1], axis=1)
		elif pad_strategy == 'zero':
			ret[:,:res.shape[1],:,:] = res
	# pdb.set_trace()
	return ret