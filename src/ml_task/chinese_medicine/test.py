# import os
# import numpy as np
# import matplotlib.pyplot as plt
#
#
# def draw_process(version, title, color, iters, data, label):
#     plt.title(title)
#     plt.xlabel("Iters")
#     plt.ylabel(label)
#     plt.cla()
#     plt.plot(iters, data, color=color, label=label)
#     dir_path = './figure/{}'.format(version)
#     if not os.path.exists(dir_path):
#         os.makedirs(dir_path)
#     plt.savefig('{}/{}.jpg'.format(dir_path, title))
#
#
# x = np.arange(0, 100)
# y = np.random.uniform(10, 100, 100)
#
# draw_process('test1', "test_title_0", "red", x, y, "test_label_0")
#
# x = np.arange(0, 50)
# y = np.random.uniform(10, 100, 50)
#
# draw_process('test1', "test_title_1", "blue", x, y, "test_label_1")
#
#
# class A:
#     def __init__(self):
#         print("class A...")
#
#     def forward(self):
#         print("A forward...")
#
#
# class B(A):
#     def __init__(self):
#         super(B, self).__init__()
#         print("class B...")
#
#     def forward(self):
#         super(B, self).forward()
#         print("B forward...")
#
#
# a = B()
# a.forward()
#
import Augmentor


p = Augmentor.Pipeline('./data/Chinese Medicine Infer', '.', 'jpg')
p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
p.zoom(probability=0.5, min_factor=1.1, max_factor=1.5)
# p.sample(100)
p.process()















