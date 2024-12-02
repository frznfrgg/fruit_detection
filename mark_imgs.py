import matplotlib.pyplot as plt
import matplotlib.image as img
from matplotlib.patches import Rectangle
import numpy as np
import glob
import re
import os


path = os.getcwd()
class_label = int(input("Введи номер класса, который хочешь разметить (0 или 1)"))


names = glob.glob(f'{path}/resized_fruits/{class_label}/*.jpg')
total_names = len(names)
N = 0

data = []
pos = []
def press(event):
	global pos, class_label, ax, N
	if event.key == ' ':
		if len(pos)==2:
			w = pos[1][0]-pos[0][0]
			h = pos[1][1]-pos[0][1]

			x = pos[0][0]+w/2
			y = pos[0][1]+h/2

			index = int(names[N].split('/')[-1][:-4])
			data.append([index,class_label,w,h,x,y])

			# draw next image
			if N+1 != total_names:
				N += 1
				image_name = names[N]
				ax.clear()
				ax.imshow(img.imread(image_name))
				plt.draw()
			else:
				print("DONE")
				np_data = np.array(data)
				np.save(f"dataset_of_class_{class_label}", np_data)
				plt.close()
		else:
			print("add bouth points first")

	# if event.key == 'enter':
	# 	np_data = np.array(data)
	# 	np.save(f"dataset_of_class_{class_label}", np_data)
	# 	plt.close()

def show_bar(pos):
	ax.add_patch(Rectangle((pos[0][0],pos[0][1]),pos[1][0]-pos[0][0], pos[1][1]-pos[0][1],  
		edgecolor='red' , facecolor='none' , lw=0.5))

	fig.canvas.draw()

def on_click(event):
	global pos, N
	if len(pos) == 2:
		pos = []
		pos.append([event.xdata, event.ydata])

		ax.clear()
		ax.imshow(img.imread(names[N]))
		plt.show()

	elif len(pos) == 0:
		pos.append([event.xdata, event.ydata])

	elif len(pos) == 1:
		pos.append([event.xdata, event.ydata])
		show_bar(pos)

fig, ax = plt.subplots()
fig.canvas.mpl_connect('key_press_event', press)
cid = fig.canvas.mpl_connect('button_press_event', on_click)

ax.imshow(img.imread(names[N]))

plt.show()

