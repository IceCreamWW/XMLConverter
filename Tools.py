import os
import shutil

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
			