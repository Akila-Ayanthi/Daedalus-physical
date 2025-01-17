import os
import sys
# supress tensorflow logging other than errors
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.insert(0, '..')

# import keras
import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
import random as rd
from tensorflow.python import debug as tf_debug
from tensorflow.keras.models import Model
from tensorflow.keras import losses

from tensorflow.keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping

from YOLOv3.model.yolo_model import YOLO
import cv2
import matplotlib.pyplot as plt
from skimage import io
import time
from tqdm import tqdm

# Parameter settings:
GPU_ID = 0							# which gpu to used
EOT_NUM = 10
CONFIDENCE = 0.3					# the confidence of attack
EXAMPLE_NUM = 10					# total number of adversarial example to generate.
BATCH_SIZE = 1						# number of adversarial example generated in each batch

BINARY_SEARCH_STEPS = 5     		# number of times to adjust the constsant with binary search
INITIAL_consts = 1		        	# the initial constsant c to pick as a first guess
CLASS_NUM = 80						# 80 for COCO dataset
MAX_ITERATIONS = 1000      		# number of iterations to perform gradient descent
ABORT_EARLY = True          		# if we stop improving, abort gradient descent early
LEARNING_RATE = 1e-2        		# larger values converge faster to less accurate results
IMAGE_SHAPE = (416, 416, 3)         # input image shape
PERT_SHAPE = (100, 100, 3)
SAVE_PATH = 'physical_examples/'
# select GPU to use
os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(GPU_ID)


def process_image(img):
	"""
	Resize, reduce and expand image.
	# Argument:
		img: original image.

	# Returns
		image: ndarray(64, 64, 3), processed image.
	"""
	image = cv2.resize(img, (416, 416),
					   interpolation=cv2.INTER_CUBIC)
	image = np.array(image, dtype='float32')
	image /= 255.
	image = np.expand_dims(image, axis=0)
	return image


def process_yolo_output(out, anchors, mask):
	"""
	Tensor op: Process output features.
	# Arguments
		out - tensor (N, S, S, 3, 4+1+80), output feature map of yolo.
		anchors - List, anchors for box.
		mask - List, mask for anchors.
	# Returns
		boxes - tensor (N, S, S, 3, 4), x,y,w,h for per box.
		box_confidence - tensor (N, S, S, 3, 1), confidence for per box.
		box_class_probs - tensor (N, S, S, 3, 80), class probs for per box.
	"""
	batchsize, grid_h, grid_w, num_boxes = map(int, out.shape[0:4])

	box_confidence = tf.sigmoid(out[..., 4:5], name='objectness')  # (N, S, S, 3, 1)
	box_class_probs = tf.sigmoid(out[..., 5:], name='class_probs')  # (N, S, S, 3, 80)

	anchors = np.array([anchors[i] for i in mask]) # Dimension of the used three anchor boxes [[x,x], [x,x], [x,x]].
	# duplicate to shape (batch, height, width, num_anchors, box_params).
	anchors = np.repeat(anchors[np.newaxis, :, :], grid_w, axis=0)          # (S, 3, 2)
	anchors = np.repeat(anchors[np.newaxis, :, :, :], grid_h, axis=0)       # (S, S, 3, 2)
	anchors = np.repeat(anchors[np.newaxis, :, :, :, :], batchsize, axis=0) # (N, S, S, 3, 2)
	anchors_tensors = tf.constant(anchors, dtype=tf.float32, name='anchor_tensors')

	box_xy = tf.sigmoid(out[..., 0:2], name='box_xy') # (N, S, S, 3, 2)
	box_wh = tf.identity(tf.exp(out[..., 2:4]) * anchors_tensors, name='box_wh') # (N, S, S, 3, 2)

	col = np.tile(np.arange(0, grid_w), grid_w).reshape(-1, grid_w)
	row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_h)

	col = col.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
	row = row.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
	grid = np.concatenate((col, row), axis=-1) #(13, 13, 3, 2)
	grid_batch = np.repeat(grid[np.newaxis, :, :, :, :], batchsize, axis=0)
	box_xy += grid_batch
	box_xy /= (grid_w, grid_h)
	box_wh /= (416, 416)
	box_xy -= (box_wh / 2.)

	# boxes -> (N, S, S, 3, 4)
	boxes = tf.concat([box_xy, box_wh], axis=-1)
	boxes = tf.reshape(boxes, [batchsize, -1, boxes.shape[-2], boxes.shape[-1]], name='boxes') #(N, S*S, 3, 4)
	# box_confidence -> (N, S, S, 3, 1) or 26 or 52
	# box_class_probs -> (N, S, S, 3, 80)
	box_confidence = tf.reshape(box_confidence, [batchsize,
												 -1,
												 box_confidence.shape[-2],
												 box_confidence.shape[-1]], name='box_confidence')
	box_class_probs = tf.reshape(box_class_probs, [batchsize,
												   -1,
												   box_class_probs.shape[-2],
												   box_class_probs.shape[-1]], name='class_probs')
	return boxes, box_confidence, box_class_probs


def process_output(raw_outs):
	"""
	Tensor op: Extract b, c, and s from raw outputs.
	# Args:
		raw_outs - Yolo raw output tensor list [(N, 13, 13, 3, 85), (N, 26, 26, 3, 85), (N, 26, 26, 3, 85)].
	# Returns:
		boxes - Tensors. (N, 3549, 3, 4), classes: (N, 3549, 3, 1), scores: (N, 3549, 3, 80)
	"""
	masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
	anchors = [[10, 13], [16, 30], [33, 23], 
			   [30, 61], [62, 45], [59, 119], 
			   [116, 90], [156, 198], [373, 326]]
	boxes, objecness, scores = [], [], []

	for out, mask in zip(raw_outs, masks):
		# out -> (N, 13, 13, 3, 85)
		# mask -> one of the masks
		# boxes (N, 13X13, 3, 4), box_confidence (N, 13X13, 3, 1)
		# box_class_probs (13X13, 3, 80) | 26 X 26 |
		b, c, s = process_yolo_output(out, anchors, mask)
		if boxes == []:
			boxes = b
			objecness = c
			scores = s
		else:
			boxes = tf.concat([boxes, b], 1, name='xywh') 
			objecness = tf.concat([objecness, c], 1, name='objectness')
			scores = tf.concat([scores, s], 1, name='class_probs')
	return boxes, objecness, scores


def pdist(xy):
	"""
	Tensor op: Computes pairwise distance between each pair of points
	# Args:
	  xy - [N,2] matrix representing N box position coordinates
	# Content:
	  dists - [N,N] matrix of (squared) Euclidean distances
	# Return:
	  expectation of the Euclidean distances
	"""
	xy2 = tf.reduce_sum(xy * xy, 1, True)
	dists = xy2 - 2 * tf.matmul(xy, tf.transpose(xy)) + tf.transpose(xy2)
	return tf.reduce_mean(dists)

def output_to_pdist(bx, by):
	"""
	Tensor op: calculate expectation of box distance given yolo outpput bx & by.
	# Args:
		bx - YOLOv3 output batch x coordinates in shape (N, 3549, 3, 1)
		by - YOLOv3 output batch y coordinates in shape (N, 3549, 3, 1)
	"""
	bxby = tf.concat([bx, by], axis=-1)
	bxby = tf.reshape(bxby, [-1, 2])
	return pdist(bxby)

def pairwise_IoUs(bs1, bs2):
	"""
	Tensor op: Calculate pairwise IoUs given two sets of boxes.
	# Arguments:
		bs1, bs2 - tensor of boxes in shape (?, 4)
	# Content:
		X11,y11------x12,y11     X21,y21------x22,y21
		 |		 	 	|			|		 	 |
		 |		 	 	|			|		 	 |
		x11,y12-------x12,y12    x21,y22-------x22,y22
	# Returns:
		iou - a tensor of the matrix containing pairwise IoUs, in shape (?, ?)
	"""
	x11, y11, w1, h1 = tf.split(bs1, 4, axis=1)  # (N, 1)
	x21, y21, w2, h2 = tf.split(bs2, 4, axis=1)  # (N, 1)
	x12 = x11 + w1
	y12 = y11 + h1
	x22 = x21 + w2
	y22 = y21 + h2
	xA = tf.maximum(x11, tf.transpose(x21))
	yA = tf.maximum(y11, tf.transpose(y21))
	xB = tf.minimum(x12, tf.transpose(x22))
	yB = tf.minimum(y12, tf.transpose(y22))
	# prevent 0 area
	interArea = tf.maximum((xB - xA + 1), 0) * tf.maximum((yB - yA + 1), 0)

	boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
	boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)

	iou = interArea / (boxAArea + tf.transpose(boxBArea) - interArea)
	print('iou', iou)
	return iou


def expectation_of_IoUs(boxes):
	"""
	Tensor op: Calculate the expectation given all pairwise IoUs.
	# Arguments
			boxes - boxes of objects. It takes (?, 4) shaped tensor;
	# Returns
			expt - expectation of IoUs of box pairs. Scalar tensor.
	"""
	IoUs = pairwise_IoUs(boxes, boxes)
	expt = tf.reduce_mean(IoUs)
	return expt

def expectation_of_IoUs_accross_classes(boxes, box_scores):
	"""
	Tensor op: Calculate IoU expectation for IoU expectations from different class.
	Arguments:
		#boxes - (3549, 3, 4) tensor output from yolo net
		#box_scores - (N1**2+N2**2+N3**2, 3, 80) tensor
	Content:
		#box_classes - (N1**2+N2**2+N3**2, 3, 1) tensor
	Returns:
		#expt_over_all_classes - The IoU expectation of box pairs over all classes.
	"""
	box_classes = tf.cast(tf.argmax(box_scores, axis=-1), tf.int32, name='box_classes')
	class_counts = tf.bincount(box_classes)
	dominating_cls = tf.argmax(class_counts)
	dominating_cls = tf.cast(dominating_cls, tf.int32)
	index = tf.equal(box_classes, dominating_cls)
	index = tf.cast(index, tf.int32)
	others, dominating_boxes = tf.dynamic_partition(boxes, index, num_partitions=2, name='dynamic_partition')
	expt_over_all_classes = expectation_of_IoUs(dominating_boxes)
	return expt_over_all_classes

class Daedalus:
	"""
	Daedalus adversarial example generator based on the Yolo v3 model.
	"""
	def __init__(self, sess, model, pert_shape=PERT_SHAPE, img_shape=IMAGE_SHAPE, batch_size=BATCH_SIZE, confidence=CONFIDENCE, du_num=EOT_NUM,
				 learning_rate=LEARNING_RATE, binary_search_steps=BINARY_SEARCH_STEPS, max_iterations=MAX_ITERATIONS,
				 abort_early=ABORT_EARLY, initial_consts=INITIAL_consts, boxmin=0, boxmax=1):
		self.sess = sess
		self.LEARNING_RATE = learning_rate
		self.MAX_ITERATIONS = max_iterations
		self.BINARY_SEARCH_STEPS = binary_search_steps
		self.ABORT_EARLY = abort_early
		self.initial_consts = initial_consts
		self.batch_size = batch_size
		self.du_num = du_num
		self.repeat = binary_search_steps >= 6
		self.yolo_model = model
		self.confidence = confidence

		def transform_perturbation(pert, img):
			"""
			generate a mask for constrainting perturbations on to an object
			Arguments:
				# pert: a perturbation tensor
				# img: an image containing an object to perturb (1,W,W,C)
			Return:
				# a transformed perturbation
			"""
			def scale(pert, img):
				'''
				Scale masks
				'''
				with tf.name_scope('scale'):
					W = tf.cast(tf.shape(img)[-2], tf.float32)
					perturb_size = tf.cast(tf.shape(pert)[-2], tf.float32)
					newscale = tf.random.uniform((), tf.minimum(0.9*perturb_size, W), tf.minimum(1.1*perturb_size, W))
					_to_size = tf.cast(newscale, tf.int32)
					return tf.image.resize_images(pert, [_to_size, _to_size])

			def rotates(pert):
				'''
				Rotate masks
				'''
				with tf.name_scope('rotate'):
					angles = np.pi * tf.random.uniform((), -0.1, 0.1)
					return tf.contrib.image.rotate(pert, angles, name='rotated_imgs')

			def pad_n_shift(pert, img):
				'''
				Shift and pad perturbation into img size
				'''
				with tf.name_scope('shift'):
					W = tf.shape(img)[-2]
					perturb_size = tf.shape(pert)[-2]
					# set positions of the perturbation according to a uniform distribution
					left = tf.random.uniform((), 0, W-perturb_size, dtype=tf.int32)
					top = tf.random.uniform((), 0, W-perturb_size, dtype=tf.int32)
					right = left + perturb_size
					bottom = top + perturb_size
					print(left, right, top, bottom, W)
					pads = tf.stack([[0,0],tf.stack([left, W-right]),tf.stack([top, W-bottom]),[0,0]])
					print(pads)
					return tf.pad(pert, pads)

			with tf.name_scope('generate_mask'):
				W = tf.shape(img)[-2]
				C = tf.shape(img)[-1]
				transformed_mask = rotates(scale(pert, img))
				return pad_n_shift(transformed_mask, img)

		def NPS(imgs):
			return

		# the perturbation we're going to optimize:
		with tf.name_scope('inputs'):
			self.perturbation = tf.Variable(np.zeros((1,
											     pert_shape[0],
											     pert_shape[1],
											     pert_shape[2])), dtype=tf.float32, name='perturbation')

			# tf variables to sending data to tf:
			self.timgs = tf.Variable(np.zeros((batch_size,
											   img_shape[0],
											   img_shape[1],
											   img_shape[2])), dtype=tf.float32, name='self.timgs')
			transformed_pertbations = transform_perturbation(self.perturbation, self.timgs)
				
			self.consts = tf.Variable(np.zeros(batch_size), dtype=tf.float32, name='self.consts')

			# and here's what we use to assign them:
			self.assign_timgs = tf.placeholder(tf.float32, (batch_size,
															img_shape[0],
															img_shape[1],
															img_shape[2]))
			self.assign_consts = tf.placeholder(tf.float32, [batch_size])

			# Tensor operation: the resulting image, tanh'd to keep bounded from
			# boxmin to boxmax:
			self.boxmul = (boxmax - boxmin) / 2.
			self.boxplus = (boxmin + boxmax) / 2.
			self.newimgs = tf.tanh(transformed_pertbations + self.timgs) * self.boxmul + self.boxplus

			# Get prediction from the model:
			outs = self.yolo_model._yolo(self.newimgs)
			# [(N, 13, 13, 3, 85), (N, 26, 26, 3, 85), (N, 52, 52, 3, 85)]
			print(outs)
			# (N, 3549, 3, 4), (N, 3549, 3, 1), (N, 3549, 3, 80)
			boxes, objectness, classprobs = process_output(outs)
			self.bx = boxes[..., 0:1]
			self.by = boxes[..., 1:2]
			self.bw = boxes[..., 2:3]
			self.bh = boxes[..., 3:4]
			self.obj_scores = objectness
			self.class_probs = classprobs
			self.box_scores = tf.multiply(self.obj_scores, tf.reduce_max(self.class_probs, axis=-1, keepdims=True))

		with tf.name_scope('calc_loss'):
			# Optimisation metrics:
			self.l2dist = tf.reduce_sum(tf.square(self.newimgs - (tf.tanh(self.timgs) * self.boxmul + self.boxplus)), [1, 2, 3])
			print('self.l2dist', self.l2dist)
			# Define DDoS losses: loss must be a tensor here!
			# Make the objectness of all detections to be 1.
			self.loss1_1_x = tf.reduce_mean(tf.square(self.box_scores - 1), [-3, -2, -1])
			print('self.loss1_1_x', self.loss1_1_x)
			# Minimising the size of all bounding box.
			#self.f1 = tf.reduce_mean(IoU_expts)
			self.f3 = tf.reduce_mean(tf.square(tf.multiply(self.bw, self.bh)), [-3, -2, -1])
			print('self.f3', self.f3)

			# add two loss terms together
			self.loss_adv = self.loss1_1_x + self.consts * self.f3
			self.loss1 = tf.reduce_mean(self.loss_adv)
			#self.loss2 = tf.reduce_mean(self.l2dist)
			self.loss = self.loss1# + self.loss2
		
		# Setup the adam optimizer and keep track of variables we're creating
		start_vars = set(x.name for x in tf.global_variables())
		optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE)
		self.train = optimizer.minimize(self.loss, var_list=[self.perturbation])
		end_vars = tf.global_variables()
		new_vars = [x for x in end_vars if x.name not in start_vars]

		# these are the variables to initialize when we run
		self.setup = []
		self.setup.append(self.timgs.assign(self.assign_timgs))
		self.setup.append(self.consts.assign(self.assign_consts))
		self.init = tf.variables_initializer(var_list=new_vars)
		self.reset_perturbation = tf.variables_initializer(var_list=[self.perturbation])
	
	def attack_batch(self, imgs):
		"""
		Run the attack on a batch of images and labels.
		"""

		def check_success(loss, init_loss):
			"""
			Check if the initial loss value has been reduced by 'self.confidence' percent
			"""
			return loss <= init_loss * (1 - self.confidence)

		batch_size = self.batch_size

		# convert images to arctanh-space
		imgs = np.arctanh((imgs - self.boxplus) / self.boxmul * 0.999999)

		# set the lower and upper bounds of the constsant.
		lower_bound = np.zeros(batch_size)
		consts = np.ones(batch_size) * self.initial_consts
		upper_bound = np.ones(batch_size) * 1e10

		# store the best l2, score, and image attack
		o_bestl2 = [1e10] * batch_size
		o_bestloss = [1e10] * batch_size
		o_bestattack = [np.zeros(imgs[0].shape)] * batch_size
		self.sess.run(self.reset_perturbation)

		for img_ind in range(imgs.shape[0]):
			batch = imgs[img_ind:img_ind+1]

			for outer_step in range(self.BINARY_SEARCH_STEPS):
				# completely reset adam's internal state.
				self.sess.run(self.init)

				# cache the current best l2 and score.
				bestl2 = [1e10] * batch_size
				# bestconfidence = [-1]*batch_size
				bestloss = [1e10] * batch_size

				# The last iteration (if we run many steps) repeat the search once.
				if self.repeat and outer_step == self.BINARY_SEARCH_STEPS - 1:
					consts = upper_bound

				# set the variables so that we don't have to send them over again.
				self.sess.run(self.setup, {self.assign_timgs: batch,
										   self.assign_consts: consts})

				# start gradient descent attack
				print('adjust c to:', sess.run(self.consts))
				init_loss = sess.run(self.loss)
				init_adv_losses = sess.run(self.loss_adv)
				prev = init_loss * 1.5
				for iteration in range(self.MAX_ITERATIONS):
					# perform the attack on a single example
					_, l, l2s, l1s, nimgs, c, pertb = self.sess.run([self.train, self.loss, self.l2dist, self.loss_adv, self.newimgs, self.consts, self.perturbation])
					# print out the losses every 10%
					if iteration % (self.MAX_ITERATIONS // 10) == 0:
						print('\n===iteration:', iteration, '===')
						print('\nattacked box number:', sess.run(self.bw).shape)
						print('\nloss values of box confidence and dimension:', sess.run([self.loss1_1_x, self.f3]))
						print('\nadversarial losses:', l1s)
						print('\ndistortions:', l2s)

					# check if we should abort search if we're getting nowhere.
					if self.ABORT_EARLY and iteration % (self.MAX_ITERATIONS // 10) == 0:
						if l > prev * .9999:
							break
						prev = l
					# update the best result found so far
					for e, (l1, l2, ii) in enumerate(zip(l1s, l2s, pertb)):
						#print(l2, bestl2[e], l1, init_adv_losses[e])
						if l2 < bestl2[e] and check_success(l1, init_adv_losses[e]):
							bestl2[e] = l2
							bestloss[e] = l1
						if l2 < o_bestl2[e] and check_success(l1, init_adv_losses[e]):
							o_bestl2[e] = l2
							o_bestloss[e] = l1
							o_bestattack[e] = ii

				# adjust the constsant as needed
				for e in range(batch_size):
					if check_success(l1s[e], init_adv_losses[e]):
						# success, divide consts by two
						upper_bound[e] = min(upper_bound[e], consts[e])
						if upper_bound[e] < 1e9:
							consts[e] = (lower_bound[e] + upper_bound[e]) / 2
					else:
						# failure, either multiply by 10 if no solution found yet
						#          or do binary search with the known upper bound
						lower_bound[e] = max(lower_bound[e], consts[e])
						if upper_bound[e] < 1e9:
							consts[e] = (lower_bound[e] + upper_bound[e]) / 2
						else:
							consts[e] *= 10
		# return the best solution found
		o_bestl2 = np.array(o_bestl2)
		return o_bestattack, o_bestl2


	def attack(self, imgs):
		"""
		Perform the L_2 attack on the given images for the given targets.
		If self.targeted is true, then the targets represents the target labels.
		If self.targeted is false, then targets are the original class labels.
		"""
		r = []
		ds = []
		print('go up to', len(imgs))
		for i in range(0, len(imgs), self.batch_size):
			print('tick', i)
			X_adv, dists = self.attack_batch(imgs[i:i + self.batch_size])
			path = SAVE_PATH+'{0} confidence'.format(self.confidence)
			if not os.path.exists(path):
				os.makedirs(path)
			np.save(path+'/Distortions of images {0} to {1}.npy'.format(i, i+self.batch_size), dists)
			for j in range(len(X_adv)):
				io.imsave(path+'/Perturbation of {1} Distortion {2}.png'.format(self.confidence, i+j, dists[j]), X_adv[j])			
			r.extend(X_adv)
			ds.extend(dists)
		return np.array(r), np.array(ds)


if __name__ == '__main__':
	sess = tf.InteractiveSession()
	init = tf.global_variables_initializer()
	sess.run(init)
	ORACLE = YOLO(0.6, 0.5)  # The auguments do not matter.
	X_test = []
	# for (root, dirs, files) in os.walk('/home/dissana8/LAB/Visor'):
	for (root, dirs, files) in os.walk('/home/dissana8/Daedalus-physical/datasets/COCO/val2017/val2017/Test/'):
		for name in files:
			if name.endswith('jpg'):
            	# fpaths.append(os.path.join(path, name))
		# if files:
		# 	for f in files:
		# 		print(f)
				path = os.path.join(root, name)
				image = cv2.imread(path)
				image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # RGB
				image = process_image(image)
				X_test.append(image)
				EXAMPLE_NUM -= 1
				if EXAMPLE_NUM == 0:
					break
	X_test = np.concatenate(X_test, axis=0)
	print('X_test shape:', X_test.shape)
	attacker = Daedalus(sess, ORACLE)
	X_adv, distortions = attacker.attack(X_test)
	io.imsave('adv_poster.png', X_adv)
	writer = tf.summary.FileWriter("log", sess.graph)
	writer.close()