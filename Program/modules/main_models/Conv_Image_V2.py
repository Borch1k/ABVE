import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses
import cv2 as cv
import gc

def get_base_model_save_name():
	return 'CIV2'


class Model:
	"""
	Модель на базе чистых конволюций, без транспонирования данных
	
	Название модели: Conv_Image_V1
	"""
	def __init__(self, frames_number: int = 11, step: int = 5, frame_size: tuple = (270, 480, 3), learning_rate: float = 1e-5):
		"""
		Создаёт и компилирует модель

		frames_number - количество подряд идущих кадров на основе которых модель будет предсказывать класс действия
		step - шаг который будет осуществляться между началами двух наборов подряд идущих кадров должен быть в диапазоне [1;frames_number]
			то есть в случае окна 5 и шага 3 и 2 итераций интерференса, будут выбраны кадры с номерами: [0, 1, 2, 3, 4] и [3, 4, 5, 6 ,7]
		framse_size - розмер изображения в формате (высота, ширина, 3), где 3 - количество цветовых каналов, по стандарту RGB
		learning_rate - скорость обучения модели 
		"""
		self.frames_number = frames_number
		self.step = step
		assert step > 0, "step не может быть меньше 1"
		assert step < frames_number+1, "step не может быть больше, чем frames_number"
		self.frame_size = frame_size
		self.mega_frame_size = (frames_number, frame_size[0], frame_size[1], frame_size[2])

		self.model = models.Sequential([
			tf.keras.Input(shape=self.mega_frame_size),
			layers.Rescaling(scale=1/.256),
			layers.Conv3D(frame_size[2]*frames_number, (frames_number,1,1), padding='valid', activation='tanh'),
			layers.Reshape((frame_size[0], frame_size[1], frame_size[2]*frames_number)),
			layers.MaxPooling2D(),
			layers.Conv2D(80, 3, padding='same', activation='tanh'),
			layers.MaxPooling2D(),
			layers.Conv2D(160, 3, padding='same', activation='tanh'),
			layers.MaxPooling2D(),
			layers.Conv2D(160, 3, padding='same', activation='tanh'),
			layers.MaxPooling2D(),
			layers.Conv2D(320, 3, padding='same', activation='tanh'),
			layers.MaxPooling2D(),
			layers.Conv2D(320, 3, padding='same', activation='tanh'),
			layers.MaxPooling2D(),
			layers.Flatten(),
			layers.Dense(1024, activation='sigmoid'),
			layers.Dense(5, activation='softmax')
		])

		self.model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate),
			loss=losses.CategoricalCrossentropy(),
			metrics=['accuracy'])


	def train(self, frames: np.ndarray, labels: np.ndarray, batch_size: int = 16, epochs: int = None, batch_retrains: int = 5, verbose: int = 1):
		"""
		Функция для обучения модели

		frames - массив кадров, имеет размер (n, *frame_size)
		labels - массив классов по кадрам, имеет размер (n, )
		batch_size - количество датапоинтов для шага обучения
		epochs - количество эпох обучения 
			одна эпоха: выбор новых датапоинтов и обучение на них
		batch_retrains - количество раз обучения на одном наборе датапоинтов  
		verbose:
			0 - отсутсвие вывода информации об обучении
			1 - вывод информации по эпохам
			2 - вывод информации по шагам
		"""
		class_count = np.unique(labels).shape[0]
		if not epochs:
			epochs = round(frames.shape[0]*0.5/self.frames_number/batch_size)
			print(epochs)
			epochs = 10
		for e in range(epochs):
			X = []
			index = np.random.randint(0, frames.shape[0]-self.frames_number, batch_size)
			for j in range(batch_size):
				X.append(np.array(frames[index[j]:(index[j]+self.frames_number)]))
			X = np.array(X, dtype='i1')
			y = tf.one_hot((labels[index+self.frames_number//2]).reshape(batch_size,), depth=class_count)
			self.model.fit(X, y, epochs=batch_retrains, verbose=verbose)
			gc.collect()
		return self
	

	def save_model(self, filename: str):
		"""
		Функция для сохранения состояния и весов модели
		Функция ожидает, что пусть является валидным и не требует дополнительного создания директорий
			для сохранения используется формат SavedModel, который является форматом файла по умолчанию в TF2.x

		filename - путь и название файла, в котором будет сохранена модель
			стандартно название файла не требует расширения, но для удобства лучше указывать расширение '.model'
		"""
		self.model.save(filename)


	def load_model(self, filename: str):
		"""
		Функция для загрузки состояния и весов модели
		Функция ожидает, что пусть является валидным и файл модели существует
			для загрузки используется формат SavedModel, который является форматом файла по умолчанию в TF2.x

		filename - путь и название файла, из которого будет загружена модель
		"""
		self.model = tf.keras.models.load_model(filename)
		return self


	def test_model(self,  frames: np.ndarray, labels: np.ndarray, batch_size: int = 512):
		"""
		Функция для тестирования модели на всём датасете

		Функция возвращает матрицу ошибок размером (количество_классов, количество_классов)
		"""
		class_count = np.unique(labels).shape[0]
		matrix = [[0 for i in range(class_count)] for i in range(class_count)]

		for index in range(frames.shape[0]//batch_size):
			X = []
			y = []
			for j in range(batch_size):
				X.append(np.array(frames[index*batch_size+j:(index*batch_size+j+self.frames_number)], dtype='i1'))
				y.append(labels[index*batch_size+j+self.frames_number//2])
			X = np.array(X, dtype='i1')
			y = np.array(y)
			pred = np.argmax(self.model.predict(X),axis=1)
			for p,t in zip(pred, y):
				matrix[t][p] += 1
			gc.collect()
		return matrix


	def interference(self, video_filename: str, batch_size: int = 1024, verbose: int = 1):
		cap = cv.VideoCapture(video_filename)
		interfer_pred = []
		temp_frames = []
		temp_pred = []
		i = 0

		while cap.isOpened():
			ret, frame = cap.read()
			if not ret:
				print("Can't receive frame (stream end?). Exiting ...")
				break
			temp_frames.append(cv.resize(frame, (self.frame_size[1],self.frame_size[0])))
			if (len(temp_frames) == self.step*batch_size+self.frames_number):
				X = np.lib.stride_tricks\
				  .sliding_window_view(np.array(temp_frames, dtype='i1'), self.mega_frame_size)[::self.step]\
				  .reshape(batch_size+1, self.mega_frame_size[0], self.mega_frame_size[1], self.mega_frame_size[2], self.mega_frame_size[3])
				temp_pred = np.argmax(self.model.predict(X, verbose=verbose),axis=1)
				interfer_pred = np.concatenate((interfer_pred, temp_pred))
				del temp_frames[:-self.frames_number]
				gc.collect()
				i += 1
				if i == 3:
					break
		return interfer_pred