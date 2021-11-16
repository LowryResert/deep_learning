import Augmentor
import os


base_path = './data/Chinese Medicine/'
class_paths = os.listdir(base_path)
for class_path in class_paths:
    data_base_path = base_path + class_path
    p = Augmentor.Pipeline(data_base_path, '.', 'jpg')
    p.flip_random(1)
    p.process()


