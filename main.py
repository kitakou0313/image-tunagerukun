import numpy as np
from PIL import Image
import re
import glob
from tqdm import tqdm
import os

# 画像を読み込み、36枚ずつ6*6で結合して保存する
if __name__ == "__main__":
    # サムネイル生成元の画像が保存されているディレクトリへのパス
    IMAGE_DATA_PATH = "./data/images"
    # 生成したサムネイルの保存先
    THUMNAIL_PATH = "./data/thumnails"

    # 一枚の画像サイズ
    IMAGE_SIZE_HEIGHT = 64
    IMAGE_SIZE_WIDTH = 64

    # 素材画像の名前のパターン {}には通し番号が入る
    IMAGE_FILE_NAME_PATTERN = "{}.png"

    # サムネイルの一行に入る画像数
    NUMBER_IMAGES_IN_ONE_THUMNAIL_COLUMN = 6
    # サムネイルの列数
    NUMBER_IMAGES_IN_ONE_THUMNAIL_ROW = 6

    # 1サムネイルに含まれる画像数
    NUMBER_IMAGES_IN_ONE_THUMNAIL = NUMBER_IMAGES_IN_ONE_THUMNAIL_ROW * \
        NUMBER_IMAGES_IN_ONE_THUMNAIL_COLUMN

    # サムネイル保存用のディレクトリ作成
    os.makedirs(THUMNAIL_PATH, exist_ok=True)

    # 素材画像のディレクトリからファイルの一覧を取得
    files = glob.glob(IMAGE_DATA_PATH+"/*")
    image_numbers = []
    image_number_regex = r"\d+"
    for file_path in files:
        res = re.findall(image_number_regex, file_path)
        image_numbers.append(int(res[0]))

    # 画像の通し番号でソート
    image_numbers.sort()

    # 画像を読み込み
    images = []
    print("Load images")
    for image_number in tqdm(image_numbers):
        images.append(np.array(Image.open(IMAGE_DATA_PATH+"/"+IMAGE_FILE_NAME_PATTERN.format(image_number)).resize(
            size=(IMAGE_SIZE_WIDTH, IMAGE_SIZE_HEIGHT))))

    # サムネイルの生成
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
        # 1行ごとに画像を結合
        for row_image_number in range(0, len(images_included_thumnail), NUMBER_IMAGES_IN_ONE_THUMNAIL_COLUMN):
            images_in_row = images_included_thumnail[row_image_number:
                                                     row_image_number+NUMBER_IMAGES_IN_ONE_THUMNAIL_COLUMN]

            concatenated_images_list.append(
                np.concatenate(images_in_row, axis=1)
            )

        # 全行を結合して一イメージに
        thumnail = np.concatenate(concatenated_images_list)

        # 作成したサムネイルを保存
        imaged_thumnail = Image.fromarray(thumnail)
        imaged_thumnail.save(THUMNAIL_PATH+"/"+"thumnail_{}_{}".format(number,
                             number+NUMBER_IMAGES_IN_ONE_THUMNAIL-1)+".png")

    pass

    # ntgaed_vecs = {}
    # for index in tqdm(idxes):
    #     ntgaed_vecs[index] = np.squeeze(
    #         np.load("datas/ntgaed_vecs/t_hundred/x_train_sst-2_ntga_fnn_"+str(index)+".npy"))
