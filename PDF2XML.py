import os, shutil
import re
import zhon.hanzi
from pdf2image import convert_from_path
import pytesseract
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from Tools import Tools
import editdistance
import json

from pdfminer.pdfinterp import PDFResourceManager, process_pdf
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from io import StringIO
from io import open



class ResumeLine():

	TITLE = 0
	CONTENT = 1

	def __init__(self, bound, _type=CONTENT, lineno=-1):
		self.bound = bound
		self._type = _type
		self.lineno = lineno
		self.image_hsv = None
		self.image_gray = None
		self.text = None
		self.content = []
		self.is_end = True

	@property
	def bound(self):
		return self.__bound

	@bound.setter
	def bound(self, new_bound):
		self.__bound = {"left":0, "top":0,"right":-1,"bottom":-1}
		self.__bound.update(new_bound)

	def update_bound(self, new_bound):
		self.__bound.update(new_bound)

	def get_image(self, hsv):
		self.image_hsv = hsv[self.bound["top"] : self.bound["bottom"],
							self.bound["left"] : self.bound["right"]]
		return self.image_hsv

	def is_title(self):
		return self._type == ResumeLine.TITLE

	def __OCR_preprocess__(self):
		bgr = cv2.cvtColor(self.image_hsv, cv2.COLOR_HSV2BGR)
		image_gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

		# reverse color and binaryzation 
		aver = np.mean(image_gray)
		is_white = (image_gray.astype('int') - int(aver) > 0).astype(int)

		# background color should be white
		if np.sum(is_white[4, 0:30]) >= 15:
			image_gray = (is_white * 255).astype("uint8")
		else:
			image_gray = ((1 - is_white) * 255).astype("uint8")

		self.image_gray = image_gray
		return image_gray

	def detect_text(self):
		if self.image_hsv is None:
			raise ValueError("No Images Available")
		image_gray = self.__OCR_preprocess__()
		self.text = pytesseract.image_to_string(image_gray, lang="chi_sim")

		rgb = cv2.cvtColor(self.image_hsv, cv2.COLOR_HSV2BGR)
		
		if self.is_title():
			self.text = re.sub("[^\u4E00-\u9FA5]","", self.text)
		else:
			self.text = re.sub("[ \n]+","", self.text)
			pass
			# raise NotImplementedError("non-title text postprocessing not implemented")
		return self.text

	def correct_text_by(self, texts):
		texts = np.array(texts)
		distances = np.array([editdistance.eval(text, self.text) for text in texts])
		self.text = min(texts[distances == distances.min()], 
							key=lambda text : abs(len(text) - len(self.text)))
		return self.text

class ResumeImageSpliter():

	# constants
	OPTIMIZE_OFF = 0
	OPTIMIZE_ON = 1

	def __init__(self, image):
		self.image_ori = image
		self.resume_lines = []
		self.image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


	def preprocess(self, processes=["clear_tableline","sharpen"]):
		img = None
		for process in processes:
			if hasattr(self, process):
				img = getattr(self, process)()
			else:
				raise ValueError("preprocess %s not defined" % process)
		return img

	def split_by_color(self, mode=OPTIMIZE_ON):
		COLOR_WHITE, COLOR_BLACK, COLOR_OTHER = (2, 1, 0)
		h,s,v = cv2.split(self.image)
		width = h.shape[1]

		# scan every line
		cnt = 1
		start = -1
		cur_color = COLOR_WHITE
		line_color = None
		for _index, (line_h, line_v) in enumerate(zip(h, v)):
			# WHITE: (0, 0, 255)     1   1
			# BLACK: (0, 0, 0)       1   0
			# OTHER: (180, 100, 100) 0   1 / 0   0 
			line_color = int((line_h == 0).all()) * (int((line_v != 0).all()) + 1)
			# for test use
			# if line_color == COLOR_WHITE:
				# s_ = "white"
			# if line_color == COLOR_BLACK:
				# s_ = "black"
			# if line_color == COLOR_OTHER:
				# s_ = "other"
			# print(s_)
			# print((line_h == 0).all(), (line_v != 0).all())
			# line is still going on
			if cur_color == line_color:
				continue
			# line meets its end
			elif line_color == COLOR_WHITE:
				_type = None
				# a title line meets its end
				if cur_color == COLOR_OTHER:
					_type = ResumeLine.TITLE
				else:
					_type = ResumeLine.CONTENT
				# print(start, _index, _type == ResumeLine.TITLE)
				cur_color = COLOR_WHITE

				resume_line = ResumeLine(bound={"top":start, "bottom": _index}, _type=_type)
				resume_line.image_hsv = self.image[start : _index, :]
				self.resume_lines.append(resume_line)
				
			# a new line start (if line_color != COLOR_WHITE)
			else:
				if cur_color != COLOR_WHITE:
					continue
				cur_color = line_color
				start = _index
		
		return self.resume_lines
		

	def mark_titles(self):
		for resume_line in self.resume_lines:
			bound = resume_line.bound
			cv2.rectangle(self.image_ori, (bound["left"], bound["top"]), (bound["right"], bound["bottom"]), (0x0,0x0,0xFF), 2)
		return self.image_ori

	def sharpen(self):
		h,s,v = cv2.split(self.image)

		s_boolean = (s.astype(int) - 100) > 0
		v_boolean = (v.astype(int) - 100) > 0

		h = (h * (s_boolean.astype("uint8")))
		s = (s_boolean.astype(int) * 255).astype("uint8")
		v = (v_boolean.astype(int) * 255).astype("uint8")

		self.image = cv2.merge((h, s, v))
		return self.image

	def clear_tableline(self):
		rows = []
		columns = []
		step = (int)(self.image.shape[1] / 4)

		h,s,v = cv2.split(self.image)
		v = (v.astype(int) - 200) < 0
		for index, row in enumerate(v):
			for i in range(0, 4):
				if np.sum(row[i * step:(i + 1) * step]) == step:
					rows.append(index)

		for index, column in enumerate(np.transpose(v)):
			for i in range(0, 4):
				if np.sum(column[i * step : (i + 1) * step]) == step:
					columns.append(index)

		for row in rows:
			self.image[row,:] = [0,0,255]

		for column in columns:
			self.image[:,column] = [0,0,255]

		return self.image



class Resume():
	def __init__(self, pdf_path, **kwargs):
		if not os.path.isfile(pdf_path):
			raise FileNotFoundError("%s is not a file" % (pdf_path))
		
		self.resume_lines = []
		self.pdf_path = pdf_path
		self.pdf_name, _ = os.path.splitext(os.path.basename(self.pdf_path))
		self.workspaces = \
		{
			"pdf_images"  	: ".\\workspace\\%s\\pdf_images" % (self.pdf_name),
			"word_images" 	: ".\\workspace\\%s\\itermediates\\word_images" % (self.pdf_name),
			"dealed_images"	: ".\\workspace\\%s\\itermediates\\dealed_images" % (self.pdf_name),
			"result" 	  	: ".\\workspace\\%s\\result" % (self.pdf_name),
			"log"			: ".\\workspace\\%s\\log" % (self.pdf_name)
		}

		for workspace in self.workspaces.values():
			os.makedirs(workspace, exist_ok=True)
			Tools.cleanfolder(workspace)
		
		self.__pdf2txt__()
		self.__pdf2image__()
		self.resume_lines = []

	def __pdf2image__(self):
		convert_from_path(self.pdf_path, output_folder=self.workspaces["pdf_images"], fmt="jpg")
		for _index, imagename in enumerate(os.listdir(self.workspaces["pdf_images"])):
			os.rename(os.path.join(self.workspaces["pdf_images"], imagename), 
				os.path.join(self.workspaces["pdf_images"], "%d.jpg" % _index))


	def __pdf2txt__(self):
		txt = self.__readPDF__(open(self.pdf_path, 'rb'))
		self.pdf_txts = [t for t in re.sub("[ \t]+"," ",txt).split('\n') if len(t) > 0]
		self.pdf_txts_guarantee = [t for t in re.sub("[^%s\n\da-zA-Z]" % zhon.hanzi.characters,"", txt).split('\n') if len(t) > 0]
		self.pdf_title_candidates = [t for t in re.sub("[^%s\n\da-zA-Z]" % zhon.hanzi.characters,"", txt).split('\n') if len(t) > 0 and len(t) < 10]
		# print(txt)
		# print()
		# for t in self.pdf_txts:
			# print(t)

	def __readPDF__(self, pdf_file):
	    rsrcmgr = PDFResourceManager()
	    retstr = StringIO()
	    laparams = LAParams()
	    device = TextConverter(rsrcmgr, retstr, laparams=laparams)

	    process_pdf(rsrcmgr, device, pdf_file)
	    device.close()

	    content = retstr.getvalue()
	    retstr.close()
	    return content

	def get_title_lines(self):
		return [resume_line for resume_line in self.resume_lines if resume_line.is_title()]


	def get_line_bounds(self):

		for _index, resume_line in enumerate(self.resume_lines):
			# if not resume_line.is_title():
				# continue
			resume_line.lineno = _index
			h, s, v = cv2.split(resume_line.image_hsv)
			width = resume_line.image_hsv.shape[1]

			feature = s if resume_line.is_title() else v
			stds = np.array([np.std(feature[:, _]) for _ in range(width)])

			aver_std = np.mean(stds[stds > 1])

			left_bound = np.argmax(stds > aver_std * 0.3)
			right_bound = width - np.argmax(stds[::-1] > aver_std * 0.3)
			resume_line.update_bound({"left":left_bound, "right":right_bound})
			bound = resume_line.bound 
			resume_line.image_hsv = resume_line.image_hsv[:,bound["left"]:bound["right"]]
		
		return self.resume_lines
	
	def detect_titles(self, correct_by=None):
		resume_lines = self.get_title_lines()

		for resume_line in resume_lines:
			resume_line.detect_text()
			if correct_by is not None:
				resume_line.correct_text_by(correct_by)

		return [resume_line.text for resume_line in resume_lines]

	def execute(self, view_words=False, mark_titles=False):
		self.spliters = []
		self.resume_lines = []

		for _index, image_file in enumerate(os.listdir(self.workspaces["pdf_images"])):
			image = cv2.imread(os.path.join(self.workspaces['pdf_images'], image_file))
			spliter = ResumeImageSpliter(image)
			spliter.preprocess()
			spliter.split_by_color()
			self.resume_lines.extend(spliter.resume_lines)

		self.resume_lines = Tools.drop_outlets(self.resume_lines, 
							key=lambda line : (line.bound["bottom"] - line.bound["top"]))

		self.get_line_bounds()
		self.detect_titles(correct_by=self.pdf_title_candidates)
		self.detect_horizontal_bound()
		# if mark_titles:
		# 	image = spliter.mark_titles()
		# 	cv2.imwrite(os.path.join(self.workspaces["dealed_images"], "%d.jpg" % _index), image)
		
		# if view_words:
		# 	for resume_line in spliter.get_title_lines():
		# 		cv2.imwrite(os.path.join(self.workspaces["word_images"], "%s.jpg" % resume_line.text), resume_line.image_gray)

		# self.spliters.append(spliter)
		# self.resume_lines.extend(spliter.resume_lines)

	def get_titles(self):
		return [resume_line.text for resume_line in self.resume_lines if resume_line.is_title()]


	def detect_horizontal_bound(self):
		bounds = [resume_line.bound for resume_line in self.resume_lines]
		self.horizontal_bound = {
			"left": max(bounds, key=lambda bound : bound["left"])["left"],
			"right": max(bounds, key=lambda bound : bound["right"])["right"]
		}
		# print("horizontal_bound = %d" % self.horizontal_bound["right"])
		return self.horizontal_bound


	def reform_json(self):
		self.struct = {}
		cur_title = None
		is_end = True
		_index = -1

		for resume_line in self.resume_lines:
			if resume_line.is_title():
				
				new_index = self.pdf_txts_guarantee.index(resume_line.text)
				if _index < new_index - 1 and _index != -1:
					for i in range(_index + 1, new_index):
						self.struct[cur_title][-1] += " " + self.pdf_txts[i]

				_index = new_index
				cur_title = resume_line.text
				self.struct[cur_title] = []	

			else:
				if cur_title is None:
					continue
				_index += 1
				if not is_end and re.search("^\d", self.pdf_txts[_index]) is None and len(self.struct[cur_title]) != 0:
					self.struct[cur_title][-1] += self.pdf_txts[_index].strip()
				else:
					self.struct[cur_title].append(self.pdf_txts[_index].strip())
				
				if np.isclose(self.horizontal_bound["right"], resume_line.bound["right"], rtol=0.05):
					is_end = False
				else:
					is_end = True
		return self.struct


if __name__ == '__main__':
	resume = Resume(".\\XM.pdf")
	resume.execute(view_words=True, mark_titles=True)
	# d = resume.make_json()
	print(json.dumps(resume.reform_json(), indent=4, sort_keys=False, ensure_ascii=False))
	