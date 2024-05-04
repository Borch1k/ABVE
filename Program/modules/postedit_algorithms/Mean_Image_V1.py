import numpy as np

class Image_V1:
	"""
	Алгоритм постобработки основанный на оконном усреднении

	Название алгоритма постобработки Mean_Image_V1
	"""
	def __init__(self, interests: dict, window_sizes: list = [11]):
		"""
		Задаёт основные параметры алгоритма
		
		interests - dict состоящий из пар: номер_класса: интерес
			интерес - число в диапазоне [0;1]
			если интерес равен 0 - значит этот класс не надо оставлять в финальном видео
			если интерес равен 1 - значит этот класс точно надо оставить в финальном видео
			для всего оставльного используются значения между
		window_sizes - массив окон, каждое из которых будет применено последовательно
			например массив [11, 11] значит, что алгоритм усреднит интерес 
		"""
		self.window_sizes = window_sizes
		self.interests = interests


	def post_edit(self, interfer_pred):
		"""
		Применяет алгоритм постобработки на данных предоставленных моделью

		interfer_pred - ответ модели

		Возвращает значения интереса для каждого из кадров видео
		"""
		interfer_labels = np.array(interfer_pred+1, dtype=int)

		x = np.array(list(map(self.interests.get, interfer_labels)))

		for window_size in self.window_sizes:
			x = np.convolve(x, np.ones(window_size)/window_size, mode='valid')

		return x