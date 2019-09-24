# import os

# imdir = 'images'
# if not os.path.isdir(imdir):
#     os.mkdir(imdir)

# # fidget_folders = [folder for folder in os.listdir('.') if 'fidget' in folder]

# n = 0
# for folder in imdir:
#     for imfile in os.scandir(folder):
#         os.rename(imfile.path, os.path.join(imdir, '{:06}.png'.format(n)))
        # n += 1

from glob import glob
import cv2

path = glob('./**/*.jpg', recursive=True)
# path = glob('images/**/*.jpg', recursive=True)

id = 0

for file in path:
	img = cv2.imread(file)
	cv2.imwrite('train/images/' + '{:06}.png'.format(id), img)
	print("Please Wait! Processing rename file {} from {}".format(id, len(path)))
	id +=1
	
print("\nRename file Finish")