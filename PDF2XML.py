import os, shutil
from pdf2image import convert_from_path
import pytesseract
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from Tools import Tools

class ImageConverter():
	def __init__(self, img_path, output_folder):
		if not os.path.isfile(img_path):
			raise FileNotFoundError("%s is not a file" % (img_path))

		self.segments = None
		self.rectangles = None
		self.img = cv2.imread(img_path)
		self.output_folder = output_folder
		self.HSV = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
		self.H, self.S, self.V = cv2.split(self.HSV)


	def sharpen(self):
		S = (self.S.astype(int) - 100) > 0
		V = (self.V.astype(int) - 100) > 0

		self.H = (self.H * (S.astype("uint8")))
		self.S = (S.astype(int) * 255).astype("uint8")
		self.V = (V.astype(int) * 255).astype("uint8")
		self.HSV = cv2.merge((self.H, self.S, self.V))
		return self.HSV

	# 超参数step，多少个连在一起的像素同时为深色时，算是表格的一部分？
	def clear_tableline(self):
		rows = []
		columns = []
		step = (int)(self.img.shape[1] / 4)

		V = (self.V.astype(int) - 200) < 0
		for index, row in enumerate(V):
			for i in range(0, 4):
				if np.sum(row[i * step:(i + 1) * step]) == step:
					rows.append(index)

		for index, column in enumerate(np.transpose(V)):
			for i in range(0, 4):
				if np.sum(column[i * step : (i + 1) * step]) == step:
					columns.append(index)

		# print rows and columns to see what has been cleared
		# print("rows = " + str(rows))
		# print("columns = " + str(columns))
		for row in rows:
			self.HSV[row,:] = [0,0,255]

		for column in columns:
			self.HSV[:,column] = [0,0,255]

		self.H, self.S, self.V = cv2.split(self.HSV)
		return self.HSV

	def get_segments(self):
		raise NotImplementedError()

	def mark_titles(self):
		width = self.HSV.shape[1]
		self.rectangles = []
		for segment in self.segments:
			left_bound = right_bound = -1

			border_noise_drop = int((segment[1] - segment[0]) / 8)
			new_seg = (segment[0] + border_noise_drop, segment[1] - border_noise_drop)
			
			stds = [np.std(self.S[new_seg[0]:new_seg[1],i]) for i in range(width)]
			aver_std = np.mean([std for std in stds if std > 1])

			for i in range(0, width):
				if left_bound == -1 and np.std(self.S[new_seg[0]:new_seg[1],i]) > aver_std * 0.3:
					left_bound = i
				if right_bound == -1 and np.std(self.S[new_seg[0]:new_seg[1],width - i - 1]) > aver_std * 0.3:
					right_bound = width - i - 1
				if left_bound != -1 and right_bound != -1:
					break
			self.rectangles.append((new_seg[0], left_bound, new_seg[1], right_bound))

		base_index = len(os.listdir(self.output_folder))
		title_imgs = []

		rectangle_widths = [rectangle[3] - rectangle[1] for rectangle in self.rectangles]
		aver_width = (np.sum(rectangle_widths) - np.max(rectangle_widths) - np.min(rectangle_widths)) / (len(rectangle_widths) - 2)
		max_width = aver_width * 3
		min_width = aver_width / 3
		self.rectangles = [rectangle for rectangle in self.rectangles if (rectangle[3] - rectangle[1]) > min_width and (rectangle[3] - rectangle[1]) < max_width]
		for index, rectangle in enumerate(self.rectangles):
			word_img = cv2.cvtColor(self.HSV[rectangle[0]:rectangle[2], rectangle[1]:rectangle[3]], cv2.COLOR_HSV2RGB)
			word_img_gray = cv2.cvtColor(word_img, cv2.COLOR_BGR2GRAY)
			title_imgs.append(word_img_gray)
			cv2.imwrite(os.path.join(self.output_folder, "%d.jpg" % (base_index + index + 1)), word_img_gray)
			# print(rectangle)
			cv2.rectangle(self.HSV, (rectangle[1], rectangle[0]), (rectangle[3], rectangle[2]), (0x0,0xFF,0xFF), 5)

		for title_img in title_imgs:
			print(pytesseract.image_to_string(title_img, lang="chi_sim"))

		return self.HSV


class ColorBasedConverter(ImageConverter):
	def __init__(self, img_path, output_folder):
		ImageConverter.__init__(self, img_path, output_folder)

	def __get_segments__(self):
		start = -1
		cnt = 1
		segments = []
		for index,line in enumerate(self.H):
			is_blank = (line == 0).all()
			if (start == -1) and (not is_blank):
				start = index
			elif (start != -1) and (is_blank):
				segments.append((start, index, index - 1 - start, cnt))
				cnt += 1
				start = -1

		if len(segments) <= 2:
			raise ValueError("Image cannot be converted through color features")

		segment_heights = [segment[2] for segment in segments]
		counts = np.bincount(segment_heights)
		mode = np.argmax(counts)
		if np.max(counts) >= int(0.6 * (len(segment_heights))):
			segments = [segment for segment in segments if segment[2] == mode]
			return (True, segments, self.HSV)

		aver_height = (np.sum(segment_heights) - np.max(segment_heights) - np.min(segment_heights)) / (len(segment_heights) - 2)
		max_height = aver_height * 3
		min_height = aver_height / 3
		min_height_dropped_segments = [segment for segment in segments if segment[2] < min_height]
		max_height_dropped_segments = [segment for segment in segments if segment[2] > max_height]
		
		# print to see what segments do you have
		# for segment in segments:
			# print(segment)
		# print(min_dropped_segments)
		# print(max_dropped_segments)
		max_dropped_segments = [segment for segment in segments if segment[2] >= max_height]
		min_dropped_segments = [segment for segment in segments if segment[2] <= min_height]
		segments = [segment for segment in segments if segment[2] <= max_height and segment[2] >= min_height]

		self.std_height = np.mean([segment[2] for segment in segments])

		for segment in min_height_dropped_segments:
			self.HSV[segment[0]:segment[1],:] = [0,0,255]

		# reserved code for inline image elimination
		# for segment in max_dropped_segments:
		# 	status = 0
		# 	bounds = []
		# 	for row in range(self.HSV.shape[1] - 1, 0, -1):
		# 		is_blank = (np.count_nonzero(self.H[segment[0]:segment[1], row]) < max_height)
		# 		if status == 0 and not is_blank:
		# 			status = row
		# 		elif status != 0 and is_blank:
		# 			bounds.append((row, status))
		# 			status = 0
		# 	for bound in bounds:
		# 		self.HSV[segment[0]:segment[1],bound[0]:bound[1]] = [0,0,255]

		self.H, self.S, self.V = cv2.split(self.HSV)
		return ((len(min_dropped_segments) == len(min_dropped_segments) == 0), segments, self.HSV)

	def get_segments(self):
		done = False
		while not done:
			done, self.segments, HSV = self.__get_segments__()
		return (self.segments, HSV)

class FontBasedConverter(ImageConverter):
	def __init__(self, img_path, output_folder):
		ImageConverter.__init__(self, img_path, output_folder)

	def get_segments(self):
		raise NotImplementedError()



class PDF2XML():
	def __init__(self, pdf_path):
		if not os.path.isfile(pdf_path):
			raise FileNotFoundError("%s is not a file" % (pdf_path))
		self.pdf_path = pdf_path
		self.__setpaths__()
		self.__pdf2image__()
	
	def __setpaths__(self):
		self.pdf_name, ext = os.path.splitext(os.path.basename(self.pdf_path))
		self.work_dirs = {}
		self.work_dirs["pdf_images_dir"] = ".\\%s\\images\\pdf_images" % (self.pdf_name)
		self.work_dirs["words_images_dir"] = ".\\%s\\images\\words_images" % (self.pdf_name)
		for name in self.work_dirs:
			os.makedirs(self.work_dirs[name], exist_ok=True)	
			Tools.cleanfolder(self.work_dirs[name])

	def __pdf2image__(self):
		convert_from_path(self.pdf_path, output_folder=self.work_dirs["pdf_images_dir"], fmt="jpg")
		for index, imagename in enumerate(os.listdir(self.work_dirs["pdf_images_dir"])):
			os.rename(os.path.join(self.work_dirs["pdf_images_dir"], imagename), 
				os.path.join(self.work_dirs["pdf_images_dir"], "%d.jpg" % (index + 1)))

if __name__ == '__main__':
	dealer = ColorBasedConverter(".\\pictures\\1.jpg", ".\\XM\\images\\words_images")

	HSV = dealer.clear_tableline()
	HSV = dealer.sharpen()
	segments, HSV = dealer.get_segments()
	HSV = dealer.mark_titles()

	img = cv2.cvtColor(HSV, cv2.COLOR_HSV2BGR)
	cv2.imwrite("result1.jpg", img)
