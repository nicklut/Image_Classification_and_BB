"""
Author:   Nicholas Lutrzykowski
Course:   CSCI 6270
Homework:  6
File:    hw6.py
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from sklearn.metrics import confusion_matrix

from hw6_model_2022 import RCNN
from hw6_datasets_2022 import HW6Dataset, HW6DatasetTest
from evaluation import area, iou, predictions_to_detections, evaluate

import os
import math
import random
import sys
import copy
import numpy as np
import cv2
import matplotlib.pyplot as plt

def plot_loss(training_loss, validation_loss, epochs):
	plt.plot(epochs, training_loss, label='Training Loss')
	plt.plot(epochs, validation_loss, label='Validation Loss')

	plt.title("Loss vs Epoch")
	plt.ylabel("Loss")
	plt.xlabel("Epoch")
	path = os.path.join(os.getcwd(),'output_images')
	plt.savefig(os.path.join(path,'loss_graph.png'))


def loss_func(pred_class, pred_bb, class_target, bb_target, batch_size, lam):
	'''
	weights = [0.5, 1.0, 1.0, 1.0, 0.3, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
	class_weights = torch.FloatTensor(weights).cuda()
	self.criterion = nn.CrossEntropyLoss(weight=class_weights)
	'''
	weights = [0.3, 3.0, 2.0, 3.0, 0.55]
	class_weights = torch.FloatTensor(weights).cuda()

	loss_fn = nn.CrossEntropyLoss(weight=class_weights)
	loss = loss_fn(pred_class, class_target)

	class_target_temp = class_target - 1
	pred_bb_res = torch.zeros(class_target.size()[0],4).to(device)
	for i in range(class_target.size()[0]):
		if class_target_temp[i] == -1: continue
		target = class_target_temp[i]
		pred_bb_res[i] = pred_bb[i, target*4:(target*4)+4]

	 
	dif = (bb_target-pred_bb_res)**2
	dif = torch.sum(dif, 1)
	loss_bb = torch.mean(dif)


	return loss + (lam*loss_bb)

def train(dataloader, model, optimizer, batch_size, lam):
	size = len(dataloader.dataset)
	num_batches = len(dataloader)
	model.train()

	train_loss = 0 
	# candidate_region, resized_gt_bbox, self.ground_truth_classes[idx]
	# The above is the structure of the sample variable
	for batch, sample in enumerate(dataloader):
		X, y_1, y_2 = sample[0], sample[1], sample[2]
		X, y_1, y_2 = X.to(device), y_1.to(device), y_2.to(device)
		
		
		# Find prediction error 
		pred_class, pred_bb = model(X)
		loss = loss_func(pred_class, pred_bb, y_2, y_1, batch_size, lam)
		train_loss += loss.item()
		# Backpropagation 
		optimizer.zero_grad()
		loss.backward()
		optimizer.step() 

		
		if batch % 100 == 0: 
			loss, current = loss.item(), batch * len(X)
			print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

	train_loss /= num_batches 
	return train_loss
		

def test(dataloader, model, batch_size, lam):
	size = len(dataloader.dataset)
	num_batches = len(dataloader)
	model.eval()
	test_loss, correct_label, correct = 0, 0, 0
	
	predictions = np.array([])
	actual = np.array([])

	with torch.no_grad():
		for sample in dataloader:
			X, y_1, y_2 = sample[0], sample[1], sample[2]
			X, y_1, y_2 = X.to(device), y_1.to(device), y_2.to(device)

			pred_class, pred_bb = model(X)
			test_loss += loss_func(pred_class, pred_bb, y_2, y_1, batch_size, lam).item()
			correct_label += (pred_class.argmax(1) == y_2).type(torch.float).sum().item()
			
			temp1 = pred_class.argmax(1).cpu().numpy()
			temp2 = y_2.cpu().numpy()
			predictions = np.concatenate((predictions, temp1))
			actual = np.concatenate((actual, temp2)) 

			pred_res = (pred_class.argmax(1) == y_2)
			for i in range(pred_bb.size()[0]):
				if pred_res[i] and y_2[i] == 0:
					correct += 1
				elif pred_res[i]:
					target = pred_class.argmax(1)[i] - 1
					if iou(pred_bb[i, target*4:(target*4)+4], y_1[i]) > 0.5:
						correct += 1

	confusion = confusion_matrix(actual, predictions)
	#print("SUM:", np.sum(confusion))

	test_loss /= num_batches 
	correct_label /= size 
	correct /= size 
	#print(f"Test Error: \n Accuracy BB: {(100*correct):>0.1f}% \n")
	
	print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
	print("Confusion Matrix:")
	print(confusion)
	#print(f"Avg loss: {test_loss:>8f} \n")
	return correct*100, test_loss

if __name__ == '__main__':
	
	# Get cpu or gpu device for training.
	device = "cuda" if torch.cuda.is_available() else "cpu"
	print(f"Using {device} device")

	# Get the data
	data_train = HW6Dataset('./hw6_data_2022/train', './hw6_data_2022/train.json')
	data_valid = HW6Dataset('./hw6_data_2022/valid', './hw6_data_2022/valid.json')
	data_test = HW6DatasetTest('./hw6_data_2022/test', './hw6_data_2022/test.json')
	
	batch_size = 80

	train_dataloader = DataLoader(data_train, batch_size=batch_size, num_workers=3)
	test_dataloader = DataLoader(data_test, num_workers=3)
	valid_dataloader = DataLoader(data_valid, batch_size=batch_size, num_workers=3)

	#print(len(data_valid))
	'''
	for i in range(1):
		model = RCNN().to(device)
		
		#  Define the Mean Squared error loss function as the criterion for this network's training
		#loss_fn = loss()
		optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

		epochs = 1
		#print(test_dataloader[0])
		lambda_val = 4
		for t in range(epochs):
			print(f"Epoch {t+1}\n-------------------------------")
			train(train_dataloader, model, optimizer, batch_size, lambda_val)
			test(valid_dataloader, model, batch_size, lambda_val)
			print("Done!")
	'''
	
	# Training and Validation 
	
	learn_rates = np.array([1e-5, 1e-4, 1e-3])
	#learn_rates = np.array([1e-4])

	#lambdas = np.array([0.5, 1, 1.25])
	epochs = 15
	max_accuracy = 0 
	best_epoch = 0 
	best_lr = 0
	best_model = 0
	lambda_val = 2
	best_train_loss = [] 
	best_valid_loss = [] 

	
	for learn_rate in learn_rates:
		print(f"Learning Rate {learn_rate}\n-------------------------------")
		model = RCNN().to(device)
		optimizer = torch.optim.SGD(model.parameters(), lr=learn_rate)

		valid_list = np.zeros(epochs)
		
		train_loss = [] 
		valid_loss = [] 

		for ep in range(epochs): 
			print(f"Epoch {ep+1}\n-------------------------------")
			train_l = train(train_dataloader, model, optimizer, batch_size, lambda_val)
			print("Validation Loss")
			valid_list[ep], valid_l = test(valid_dataloader, model, batch_size, lambda_val)
			train_loss.append(train_l)
			valid_loss.append(valid_l)

		epoch = np.argmax(valid_list) 
		accuracy = valid_list[epoch]

		if accuracy > max_accuracy: 
			max_accuracy = accuracy
			best_epoch = epoch 
			best_lr = learn_rate
			best_model = model
			best_train_loss = train_loss 
			best_valid_loss = valid_loss
	
	#train_loss, valid_loss = train_loss.numpy(), valid_loss.numpy()
	#plot_loss(np.array(train_loss), np.array(valid_loss), np.arange(1, epochs+1))
	plot_loss(np.array(best_train_loss), np.array(best_valid_loss), np.arange(1, epochs+1))
	
	print("Selected Model:")
	print("Epochs:", best_epoch)
	print("Learning Rate:", best_lr)
	print("lambda:", lambda_val)
	
	# Test Data 
	# Post processing goes one image at a time 
	model = best_model 
	test_size = len(test_dataloader.dataset)
	mAP = 0 
	img_num = [1, 16, 28, 30, 39, 75, 78, 88, 93, 146]
	#img_num = [1]
	print("Training Data Results")
	test(train_dataloader, model, batch_size, lambda_val)
	
	with torch.no_grad():
		#Sample Structure: np.array(image), candidate_regions, self.candidate_bboxes[idx], 
		#				   self.ground_truth_bboxes[idx], self.ground_truth_classes[idx]
		count = 1
		for sample in test_dataloader:
			img, candidate_regions, candidate_bboxes, gt_bboxes, gt_classes = sample[0], sample[1], sample[2], sample[3], sample[4]
			candidate_regions, candidate_bboxes  = candidate_regions[0].to(device), candidate_bboxes[0].to(device)
			gt_bboxes, gt_classes = gt_bboxes[0].to(device), gt_classes[0].to(device)
			predictions = [] 
			gt_detections = [] 
			img = img[0].numpy()
			img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

			pred_classes, pred_bboxes = model(candidate_regions)

			
			# Convert model output to predictions 
			predictions = [] 

			for i in range(pred_bboxes.size()[0]):
				pred_bb, candidate_bbox, pred_class = pred_bboxes[0], candidate_bboxes[0], pred_classes[0]
				#print("pred_class", pred_class)
				#print("pred_bb", pred_bb.size())
				#print("candidate_bbox", candidate_bbox.size())

				pred_bb, candidate_bbox, pred_class = list(pred_bb.tolist()), list(candidate_bbox.tolist()), list(pred_class.tolist())
				# Get the predicted class 
				pred_class, pred_bb, candidate_bbox = np.array(pred_class), np.array(pred_bb), np.array(candidate_bbox)
				
				#print("pred_class",pred_class)
				class_index = np.argmax(pred_class)-1 
				
				#print("candidate_bbox", candidate_bbox)
				if class_index == -1:
					continue 

				activation = np.amax(pred_class)
				pred_bb = pred_bb[class_index*4:(class_index*4)+4]

				candidate_region_size = 224

				#candidate_region = img.crop((candidate_bbox[0].item(), candidate_bbox[1].item(), candidate_bbox[2].item(), candidate_bbox[3].item()))
				width, height = candidate_bbox[2]-candidate_bbox[0], candidate_bbox[3] - candidate_bbox[1]

				x_scale = candidate_region_size / width 
				y_scale = candidate_region_size / height

				# Convert to pixel values 
				pixel_bb_x0 = ((pred_bb[0]*candidate_region_size) / x_scale) + candidate_bbox[0]
				pixel_bb_y0 = ((pred_bb[1]*candidate_region_size) / y_scale) + candidate_bbox[1]
				pixel_bb_x1 = ((pred_bb[2]*candidate_region_size) / x_scale) + candidate_bbox[0]
				pixel_bb_y1 = ((pred_bb[3]*candidate_region_size) / y_scale) + candidate_bbox[1]
				
				pixel_bb = np.array([pixel_bb_x0, pixel_bb_y0, pixel_bb_x1, pixel_bb_y1])
				
				pred_dict = {
					"class": class_index+1,
					"rectangle": pixel_bb, 
					"a": activation
				}
				predictions.append(pred_dict)
				#print("prediction", pixel_bb)
			

			# TODO: Convert all to numpy array? 
			
			# Create dictionary of ground truth detections
			
			gt_detections = [] 
			for i in range(gt_bboxes.size()[0]):
				gt_bbox, gt_label = gt_bboxes[0], gt_classes[0]
				gt_dict = {
					"class": gt_label.item(),
					"rectangle": list(gt_bbox.tolist())
				}
				gt_detections.append(gt_dict)
			
			
			
			detections = predictions_to_detections(predictions)
			correct, incorrect, missed, ap = evaluate(detections, gt_detections)
			mAP += ap 
			label_names = ["nothing", "bicycle", "car", "motorbike","person"]
			font = cv2.FONT_HERSHEY_SIMPLEX
			if count in img_num: 
				print("Image {:d} AP is: {:.3f}".format(count, ap))
				for det in correct:
					rect = [int(x) for x in det["rectangle"]]
					label = label_names[det["class"]]
					img = cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)
					img = cv2.putText(img, label, (int((rect[2]+rect[0])/2), rect[1]), font, 1, (255,255,255), 2, cv2.LINE_AA)

				for det in incorrect: 
					rect = [int(x) for x in det["rectangle"]]
					label = label_names[det["class"]]
					img = cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), (0, 0, 255), 2)
					img = cv2.putText(img, label, (int((rect[2]+rect[0])/2), rect[1]), font, 1, (255,255,255), 2, cv2.LINE_AA)

				for det in missed: 
					rect = [int(x) for x in det["rectangle"]]
					#print("CLASSS OUTPUT", det["class"])
					label = label_names[det["class"]]
					img = cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 255), 2)
					img = cv2.putText(img, label, (int((rect[2]+rect[0])/2), rect[1]), font, 1, (255,255,255),2, cv2.LINE_AA)

				#cv2.imwrite('/path/to/destination/image.png',image)
				path = os.path.join(os.getcwd(),'output_images')

				cv2.imwrite(os.path.join(path, str(count)+'.png'), img)

			count += 1

			


	mAP /= test_size 
	print("Final mean average precision: {:.3f}".format(mAP))
	print("Success!!")


				
	
	
	'''
	Testing (For each image):

	1) Convert predicted coordinates to image coordinates 
	2) 
	'''


