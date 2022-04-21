import os
import sys

import numpy
import pandas
from PIL import Image
from skimage.color import rgb2gray
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from skimage import io, transform
from sklearn.preprocessing import MinMaxScaler



class Data:
    BASE_DIR = ""
    DATA_DIR = ""
    ORIGINAL_DIR = ""
    TRAIN_DIR = ""
    TEST_DIR = ""
    CROP_DIR=""
    TEST_CROP_DIR=""
    def __init__(self):
        self.BASE_DIR = os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)
            ))
        self.DATA_DIR = os.path.join(self.BASE_DIR, 'data')
        self.ORIGINAL_DIR = os.path.join(self.DATA_DIR, 'original')
        self.TRAIN_DIR = os.path.join(self.DATA_DIR, 'train')
        self.TEST_DIR = os.path.join(self.DATA_DIR, 'test')
        self.CROP_DIR=os.path.join(self.DATA_DIR,'crop')
        self.TEST_CROP_DIR = os.path.join(self.DATA_DIR, 'test_crop')
        self.MODEL_DIR= os.path.join(self.BASE_DIR, 'model')

    def get_data(self, set_path, by_file=False):
        name=''
        if  set_path==self.CROP_DIR:
            name='crop'
        if set_path==self.TEST_CROP_DIR:
            name='test_crop'
        if set_path==self.TRAIN_DIR:
            name='train'
        if set_path==self.TEST_DIR:
            name='test'
        if by_file:
            res=numpy.load(os.path.join(self.DATA_DIR,name+'_input.npy'))
            output = numpy.load(os.path.join(self.DATA_DIR, name + '_output.npy'))
            # mm = MinMaxScaler()
            # res=mm.fit_transform(res)
            return res, output
        dir_list = os.listdir(set_path)
        print("Load data from "+ set_path)
        size=32
        res = []
        output = []
        # for i in range(1):
        for i in range(len(dir_list)):
            # print(bcolors.OKBLUE+dir_list[i]+bcolors.ENDC)
            label = int(dir_list[i])
            path = os.path.join(set_path, dir_list[i])
            for imagePath in os.listdir(path):
                print("\r", end="")
                print('...............' + path, end="")
                sys.stdout.flush()
                img = io.imread(os.path.join(path, imagePath), as_gray=True)
                img = transform.resize(img, (size, size))
                #edges = feature.canny(img, sigma=0.6)
                # res.append(edges)
                mm = MinMaxScaler()
                im=mm.fit_transform(img)
                res.append(im)
                output.append(label)
            # print("\n")
        print("Data loading finish")
        res = numpy.array(res)
        num, nx, ny = res.shape
        res = res.reshape((num, nx * ny))
        output = numpy.array(output)
        numpy.random.seed(100)
        numpy.random.shuffle(res)
        numpy.random.seed(100)
        numpy.random.shuffle(output)
        numpy.save(os.path.join(self.DATA_DIR,name+'_input'),res)
        numpy.save(os.path.join(self.DATA_DIR,name+'_output'),output)
        return res, output

    def load_data(self,test_dir:str)->(numpy.ndarray,numpy.ndarray):
        # get folder
        test_dir_list=os.listdir(test_dir)
        csv_list=[]
        res=[]
        output=[]
        # read csv file
        for i in test_dir_list:
            label_path=os.path.join(test_dir,i)
            for j in os.listdir(label_path):
                if j.endswith('.csv'):
                    file =os.path.join(label_path,j)
                    temp=pandas.read_csv(file,sep=';').values
                    csv_list.append(temp)
        # read image file
        for i in range(len(test_dir_list)):
            print('From '+test_dir_list[i])
            test=csv_list[i]
            for item in test:
                image_path=os.path.join(os.path.join(test_dir,test_dir_list[i]),item[0])
                image=Image.open(image_path)
                # crop
                box = [int(item[3]), int(item[4]), int(item[5]), int(item[6])]
                image = image.crop(box)
                image=numpy.array(image)
                # garyscale
                image=rgb2gray(image)
                # resize
                image = transform.resize(image, (32, 32))
                # MinMaxScaler
                mm = MinMaxScaler()
                image = mm.fit_transform(image)
                res.append(image)
                output.append(int(test_dir_list[i]))
        res = numpy.array(res)
        num, nx, ny = res.shape
        res = res.reshape((num, nx * ny))
        # shuffle
        output = numpy.array(output)
        numpy.random.seed(100)
        numpy.random.shuffle(res)
        numpy.random.seed(100)
        numpy.random.shuffle(output)
        return res, output
if __name__ == '__main__':
    data=Data()
    data.get_data(set_path=data.CROP_DIR)
    data.get_data(set_path=data.TEST_CROP_DIR)
    data.get_data(set_path=data.TRAIN_DIR)
    data.get_data(set_path=data.TEST_DIR)
