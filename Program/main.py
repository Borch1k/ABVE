import numpy as np
import cv2 as cv
import os
import importlib
import argparse

def get_files_in_directory(directory):
	files = []
	for entry in os.scandir(directory):
		files.append(entry.name)
	return files


def get_dataset_sizes(filename: str):
	cap = cv.VideoCapture(filename)

	if not cap.isOpened(): 
		print("could not open :",filename)
		return

	length = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
	width  = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
	height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
	fps    = cap.get(cv.CAP_PROP_FPS)
	return length, width, height, fps


def create_dataset(filenames: list, dataset_sizes: tuple, noise_threshold: int = 10, verbose: bool = False):
	"""
	dataset_sizes - (length, height, width)
	"""
	frames = np.memmap('frames.npy', dtype='u1', mode='w+', shape=(dataset_sizes[0],dataset_sizes[1],dataset_sizes[2],3))
	labels = []

	caps = []
	for filename in filenames:
		caps.append(cv.VideoCapture(filename))
	while caps[0].isOpened():
		ret = True
		temp_frame = []
		for i in range(len(filenames)):
			ret, frame = caps[i].read()
			if not ret:
				break
			if np.max(frame) > noise_threshold:
				temp_frame = frame
				labels.append(i)
		if not ret:
			break
		frames[len(labels)-1] = cv.resize(temp_frame, (dataset_sizes[2],dataset_sizes[1]))
		if verbose and (len(labels) % 1000 == 0):
			print(len(labels))
	np.save('labels.npy',labels)
	frames.flush()


def edit_video(filename: str, interference_result: np.ndarray):
	cap = cv.VideoCapture(filename)

	fourcc = cv.VideoWriter_fourcc(*'mp4v')
	video = cv.VideoWriter('video4.mp4',fourcc,30,(1280,720))
	i = 0
	j = 0

	while cap.isOpened():
	    ret, frame = cap.read()
	    if not ret:
	        break
	    if i >= len(interference_result):
	    	break
	    sp_z = interference_result[i]
	    if sp_z > 0.5:

	      cv.putText(frame, str(sp_z),
	          (10,50), cv.FONT_HERSHEY_SIMPLEX,
	          1, (255,255,255), 1, 2)
	      video.write(frame)
	    if (i % 1000 == 0):
	      print(i)
	    i+=1
	cv.destroyAllWindows()
	video.release()

if __name__ == '__main__':
	parser = argparse.ArgumentParser("")
	parser.add_argument("interference_filename", help="Путь до видеофайла, который нужно обработать", type=str)
	args = parser.parse_args()

	dataset_files = [f'dataset/{x}' for x in get_files_in_directory('dataset')]
	length, width, height, fps = get_dataset_sizes(dataset_files[0])
	dataset_sizes = (length, 270, 480)
	if 'frames.npy' not in get_files_in_directory('.'):
		create_dataset(dataset_files, dataset_sizes)

	main_models = [importlib.import_module('modules.main_models.Conv_Image_V2')]
	saved_models = get_files_in_directory('saved_models')
	interference_result = 0
	for model in main_models:
		model_filename = model.get_base_model_save_name()
		if model_filename not in saved_models:
			frames = np.memmap('frames.npy', dtype='u1', mode='r', shape=(*dataset_sizes, 3))
			labels = np.load('labels.npy', mmap_mode='c')
			temp_model = model.Model().train(frames, labels)
			temp_model.save_model('saved_models/'+model_filename)
		temp_model = model.Model().load_model('saved_models/'+model_filename)
		interference_result = temp_model.interference(args.interference_filename, batch_size=24)
	print(interference_result, interference_result.shape)


	postedit_algorithms = [importlib.import_module('modules.postedit_algorithms.Mean_Image_V2')]
	for algorithm in postedit_algorithms:
		interference_result = algorithm.Model({1:0, 2:0.5, 3:0, 4:1, 5:0.1}).post_edit(interference_result)


	edit_video(args.interference_filename, interference_result)

