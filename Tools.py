import os
import shutil
import numpy as np

class Tools():
	@staticmethod
	def cleanfolder(folder):
		for filename in os.listdir(folder):
			filepath = os.path.join(folder, filename)
			try:
				if os.path.isfile(filepath):
					os.unlink(filepath)
				elif os.path.isdir(filepath):
					shutil.rmtree(filepath)
			except Exception as e:
				print(e)

	def drop_outlets(lst, key, _max=1.5, _min=3):
		keys = np.array([key(_) for _ in lst])
		aver = np.mean(keys)

		while True:
			keys_test =  (keys < aver * _max)
			if len(keys) - np.count_nonzero(keys_test) > 2:
				_max += 0.2
			else:
				break

		keys = (keys > aver / _min) & (keys < aver * _max)
		lst = np.array(lst)
		return lst[keys==True]