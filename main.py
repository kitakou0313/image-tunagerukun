import numpy as np
from PIL import Image
import re
import glob
from tqdm import tqdm
import os

# 画像を読み込み、36枚ずつ6*6で結合して保存する
if __name__ == "__main__":
    IMAGE_DATA_PATH = "./data/images"
    THUMNAIL_PATH = "./data/thumnails"

    # 一枚の画像サイズ
    IMAGE_SIZE_HEIGHT = 64
    IMAGE_SIZE_WIDTH = 64

    IMAGE_FILE_NAME_PATTERN = "{}.png"

    NUMBER_IMAGES_IN_ONE_THUMNAIL_ROW = 6
    NUMBER_IMAGES_IN_ONE_THUMNAIL_COLUMN = 6

    NUMBER_IMAGES_IN_ONE_THUMNAIL = NUMBER_IMAGES_IN_ONE_THUMNAIL_ROW * \
        NUMBER_IMAGES_IN_ONE_THUMNAIL_COLUMN

    os.makedirs(THUMNAIL_PATH, exist_ok=True)

    files = glob.glob(IMAGE_DATA_PATH+"/*")

    image_numbers = []
    image_number_regex = r"\d+"
    for file_path in files:
        res = re.findall(image_number_regex, file_path)
        image_numbers.append(int(res[0]))

    image_numbers.sort()

    images = []
    print("Load images")
    for image_number in tqdm(image_numbers):
        images.append(np.array(Image.open(IMAGE_DATA_PATH+"/"+IMAGE_FILE_NAME_PATTERN.format(image_number)).resize(
            size=(IMAGE_SIZE_WIDTH, IMAGE_SIZE_HEIGHT))))

    print("Generate thumnail images")
    for number in tqdm(range(0, len(images), NUMBER_IMAGES_IN_ONE_THUMNAIL)):
        images_included_thumnail = images[number:number +
                                          NUMBER_IMAGES_IN_ONE_THUMNAIL]

        # 1サムネの画像数に満たない場合は黒画像追加
        needed_dummy_image_num = NUMBER_IMAGES_IN_ONE_THUMNAIL - \
            len(images_included_thumnail)

        for _ in range(needed_dummy_image_num):
            images_included_thumnail.append(
                np.zeros_like(images_included_thumnail[0]))

        concatenated_images_list = []
        for row_image_number in range(0, len(images_included_thumnail), NUMBER_IMAGES_IN_ONE_THUMNAIL_ROW):
            images_in_row = images_included_thumnail[row_image_number:
                                                     row_image_number+NUMBER_IMAGES_IN_ONE_THUMNAIL_ROW]

            concatenated_images_list.append(
                np.concatenate(images_in_row, axis=1)
            )

        thumnail = np.concatenate(concatenated_images_list)
        imaged_thumnail = Image.fromarray(thumnail)
        imaged_thumnail.save(THUMNAIL_PATH+"/"+"thumnail_{}_{}".format(number,
                             number+NUMBER_IMAGES_IN_ONE_THUMNAIL-1)+".png")

    pass

    # ntgaed_vecs = {}
    # for index in tqdm(idxes):
    #     ntgaed_vecs[index] = np.squeeze(
    #         np.load("datas/ntgaed_vecs/t_hundred/x_train_sst-2_ntga_fnn_"+str(index)+".npy"))
