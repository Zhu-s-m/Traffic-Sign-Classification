import os
import pandas
from src.data import Data
from sklearn.model_selection import train_test_split
from PIL import Image
import shutil


def main():
    data = Data()
    original_dir_list = os.listdir(data.ORIGINAL_DIR)
    csv_list = []
    for i in original_dir_list:
        data_dir = os.path.join(data.ORIGINAL_DIR, i)
        for j in os.listdir(data_dir):
            if j.endswith('.csv'):
                file = os.path.join(data_dir, j)
                temp = pandas.read_csv(file, sep=";").values
                csv_list.append(temp)
    for i in range(len(original_dir_list)):
        train, test = train_test_split(csv_list[i], train_size=0.7, test_size=0.3)
        train_dir = os.path.join(data.TRAIN_DIR, original_dir_list[i])
        test_dir = os.path.join(data.TEST_DIR, original_dir_list[i])
        test_crop_dir=os.path.join(data.TEST_CROP_DIR, original_dir_list[i])
        if not os.path.exists(train_dir):
            os.mkdir(train_dir)
        else:
            shutil.rmtree(train_dir, ignore_errors=True)
            os.mkdir(train_dir)
        if not os.path.exists(test_dir):
            os.mkdir(test_dir)
        else:
            shutil.rmtree(test_dir, ignore_errors=True)
            os.mkdir(test_dir)
        if not os.path.exists(test_crop_dir):
            os.mkdir(test_crop_dir)
        else:
            shutil.rmtree(test_crop_dir, ignore_errors=True)
            os.mkdir(test_crop_dir)

        for item in train:
            image_path = os.path.join(os.path.join(data.ORIGINAL_DIR, original_dir_list[i]), item[0])
            image = Image.open(image_path)
            box = [int(item[3]), int(item[4]), int(item[5]), int(item[6])]
            image = image.crop(box)
            save_path = os.path.join(train_dir, item[0])
            image.save(save_path)
        for item in test:
            image_path = os.path.join(os.path.join(data.ORIGINAL_DIR, original_dir_list[i]), item[0])
            image = Image.open(image_path)
            save_path = os.path.join(test_dir, item[0])
            image.save(save_path)
            box = [int(item[3]), int(item[4]), int(item[5]), int(item[6])]
            image = image.crop(box)
            save_path = os.path.join(test_crop_dir, item[0])
            image.save(save_path)

def crop_image():
    data = Data()
    original_dir_list = os.listdir(data.ORIGINAL_DIR)
    csv_list = []
    for i in original_dir_list:
        data_dir = os.path.join(data.ORIGINAL_DIR, i)
        for j in os.listdir(data_dir):
            if j.endswith('.csv'):
                file = os.path.join(data_dir, j)
                temp = pandas.read_csv(file, sep=";").values
                csv_list.append(temp)
    for i in range(len(original_dir_list)):
        crop_list = csv_list[i]
        print(1)
        crop_dir = os.path.join(data.CROP_DIR, original_dir_list[i])
        if not os.path.exists(crop_dir):
            os.mkdir(crop_dir)
        else:
            shutil.rmtree(crop_dir, ignore_errors=True)
            os.mkdir(crop_dir)
        for item in crop_list:
            image_path = os.path.join(os.path.join(data.ORIGINAL_DIR, original_dir_list[i]), item[0])
            image = Image.open(image_path)
            # box = [int(item[3]), int(item[4]), int(item[5]), int(item[6])]
            # image = image.crop(box)
            save_path = os.path.join(crop_dir, item[0])
            image.save(save_path)


if __name__ == "__main__":
    main()
    # crop_image()
