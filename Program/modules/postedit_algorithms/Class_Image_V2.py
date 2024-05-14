import numpy as np

class Model:
	"""
	Алгоритм постобработки основанный на оконном усреднении

	Название алгоритма постобработки Mean_Image_V1
	"""
	def __init__(self, frames_number: int, step: int, interests: dict, window_sizes: list = [11]):
		"""
		Задаёт основные параметры алгоритма
		frames_number - размер окна, по которому предсказывает основная модель
		step - шаг, который основная модель использует
		interests - dict состоящий из пар: номер_класса: интерес
			интерес - число в диапазоне [0;1]
			если интерес равен 0 - значит этот класс не надо оставлять в финальном видео
			если интерес равен 1 - значит этот класс точно надо оставить в финальном видео
			для всего оставльного используются значения между
		window_sizes - массив окон, каждое из которых будет применено последовательно
			например массив [11, 11] значит, что алгоритм усреднит интерес 
		"""
		self.frames_number = frames_number
		self.step = step
		self.window_sizes = window_sizes
		self.interests = interests


	def post_edit(self, interfer_pred):
		"""
		Применяет алгоритм постобработки на данных предоставленных моделью

		interfer_pred - ответ модели

		Возвращает значения интереса для каждого из кадров видео
		"""
		# for i in range(len(interfer_pred)):


		interfer_labels_flattened = interfer_pred+1
		x = np.array(list(map(self.interests.get, interfer_labels_flattened)))
		interfer_labels = np.zeros(((x.shape[0]-1)*self.step+self.frames_number, self.frames_number), dtype=float)-1
		for i in range(self.frames_number):
			if (-self.frames_number+i+1) != 0:
				interfer_labels[i:-self.frames_number+i+1:self.step, i] = x
			else:
				interfer_labels[i::self.step, i] = x

		mean = np.mean(interfer_labels, axis=1, where=interfer_labels>-1)
		print(mean)
		return mean