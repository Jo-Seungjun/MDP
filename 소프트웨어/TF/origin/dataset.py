import tensorflow as tf
import os
import numpy as np
import random
import time

# 냐냐냐냐냐냐
# 냐냐냐냐냐냐
#데이터 불러오기
#데이터 로드하기
#배치만들기

class Data_set():
    '''
    해당 경로의 자식폴더이름을 라벨로 할당
    images
     └ cat
        └ cat1.png
        └ cat2.png
        └ cat3.png
     └ dog
        └ ddd.png
        └ asd.png
        └ asf.png
    => 'cat':image, 'cat':image, ...
    => 'dog':image, 'dog;:image, ...
    '''

    def __init__(self):
        self.folder_dir = ""
        self.file_dirs = []
        self.labels = []
        self.batch_index = 0
        self.num_categorys = 0
        self.dataset = None
        self.iterator = None

    def open(self, folder_dir, batch_size):
        if not os.path.isdir(folder_dir):
            print("Folder No Found")
        self.folder_dir = folder_dir
        categorys = os.listdir(folder_dir)

        self.num_categorys = len(categorys)
        for category in categorys:
            if not os.path.isdir(os.path.join(folder_dir,category)):
                print("File Not Found")
        
        for i in range(len(categorys)):
            one_hot = list(np.zeros(len(categorys)))
            one_hot[i] = 1
            for filename in os.listdir(os.path.join(self.folder_dir, categorys[i])):
                self.file_dirs.append(os.path.join(self.folder_dir,categorys[i],filename))
                # self.labels.append(categorys[i])
                self.labels.append(one_hot)
        self.dataset = tf.data.Dataset.from_tensor_slices((self.file_dirs,self.labels))

        def _parse_function(filename, label):
            image_string = tf.read_file(filename)
            image_decoded = tf.image.decode_jpeg(image_string)/255
            image = tf.cast(image_decoded, tf.float32)
            return image, label

        self.dataset = self.dataset.map(_parse_function)
        self.dataset = self.dataset.shuffle(1000).repeat().batch(batch_size)
        self.iterator = self.dataset.make_one_shot_iterator()

    def shuffle(self):
        i = time.time()
        random.seed(i)
        random.shuffle(self.file_dirs)
        # print(self.file_dirs)
        
        random.seed(i)
        random.shuffle(self.labels)
        # print(self.labels)
            
    def read(self):
        # images = []
        # labels = []
        # for i in range(batch):
        #     reader = open(self.file_dirs[self.batch_index],'rb')
        #     images.append(tf.Session().run(tf.image.decode_jpeg(reader.read())))
        #     reader.close()
        #     labels.append(self.labels[self.batch_index])
        #     self.batch_index = self.batch_index + 1
        #     if self.batch_index > len(self.labels): self.batch_index = 0
        # images = np.array(images)
        # labels = np.array(labels)

        images, labels = self.iterator.get_next()
        return images, labels


# Data = Data_set()
# Data.open("./face_photos")
# #print(Data.file_dirs)

# labels, images = Data.read()
# x = images
# y = labels
# # x = tf.placeholder(tf.float32,[None,224,224,3])
# # y = tf.placeholder(tf.float32,[None,7])

# net = x * tf.constant(0.8,tf.float32)
# import matplotlib.pyplot as plt

# plt.figure()
# image, label = tf.Session().run([net,y])
# print(image)
# plt.imshow(image[0])
# plt.title(label)
# plt.show()

# plt.figure()
# image, label = tf.Session().run([net,y])
# print(image)
# plt.imshow(image[0])
# plt.title(label)
# plt.show()
