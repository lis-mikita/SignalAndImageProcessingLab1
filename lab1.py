"""
Лис Н.А. РФ2
Лабораторная работа №1
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
#import pathlib
#import os
import numpy as np
link = '/Users/Niki/Documents/first_photo.jpg' #!Change link
img = tf.keras.preprocessing.image.load_img(link)
#image33 = tf.io.decode_jpeg(tf.reshape(img, []))
#print(image33)
img = tf.convert_to_tensor(np.asarray(img))

Size = img.shape
ImagesNewSize = []
decrease = (2, 4, 8, 16, 32)
print(Size[0], Size[1])

for rate in decrease:
    X = int(Size[0]/rate)
    Y = int(Size[1]/rate)
    print(X,'  ',Y)
    image = tf.image.resize(img, (X,Y))
    #image = tf.io.encode_jpeg(img,quality=80,x_density=X,y_density=Y)
    print(image.shape)
    ImagesNewSize.append(image)
    to_img = tf.keras.preprocessing.image.array_to_img(image)
    to_img.show()

'''
#print(img)

# Read images from file.
#im1 = tf.io.decode_jpeg(images_resized[1])
#print(im1)
#im2 = tf.decode_jpeg(images_resized[2])
# Compute PSNR over tf.uint8 Tensors.
#psnr1 = tf.image.psnr(im1, im2, max_val=255)
#psnr2 = tf.image.psnr(images_resized[0], img, max_val=1.0)
#print(psnr2)
#psnr = tf.image.psnr(images_resized[1], images_resized[2], max_val=255)
#with tf.Session() as sess:
#    print (sess.run(psnr))

# Typical setup to include TensorFlow.
import tensorflow as tf

# Make a queue of file names including all the JPEG images files in the relative
# image directory.
# Создайте очередь с именами файлов, включая все файлы изображений JPEG в относительной
# каталог изображений.
filename_queue = tf.train.string_input_producer(
    tf.train.match_filenames_once("./images/*.jpg"))

# Read an entire image file which is required since they're JPEGs, if the images
# are too large they could be split in advance to smaller files or use the Fixed
# reader to split up the file.
# Прочитать весь файл изображения, который требуется, так как это JPEG, если изображения
# слишком велики, их можно заранее разбить на более мелкие файлы или использовать фиксированный
# читатель, чтобы разделить файл.
image_reader = tf.WholeFileReader()

# Read a whole file from the queue, the first returned value in the tuple is the
# # filename which we are ignoring.
# Прочитать весь файл из очереди, первое возвращаемое значение в кортеже
# имя файла, которое мы игнорируем.
_, image_file = image_reader.read(filename_queue)

# Decode the image as a JPEG file, this will turn it into a Tensor which we can
# then use in training.
# Декодируйте изображение как файл JPEG, это превратит его в тензор, который мы можем
# затем использовать в обучении.
image = tf.image.decode_jpeg(image_file)

# Start a new session to show example output.
with tf.Session() as sess:
    # Required to get the filename matching to run.
    tf.initialize_all_variables().run()

    # Coordinate the loading of image files.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    # Get an image tensor and print its value.
    image_tensor = sess.run([image])
    print(image_tensor)

    # Finish off the filename queue coordinator.
    coord.request_stop()
    coord.join(threads)
'''