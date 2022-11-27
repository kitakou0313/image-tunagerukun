from torchvision import datasets
import os

# テスト用画像をディレクトリに保存する
if __name__ == "__main__":
    IMAGE_DATA_PATH = "./data/images"

    os.makedirs(IMAGE_DATA_PATH, exist_ok=True)

    data_sets = datasets.CIFAR10(
        root="./data/raw_data", train=True, download=True)

    number = 0
    for img, label in data_sets:
        save_path = IMAGE_DATA_PATH + "/" + str(number) + ".png"
        img.save(save_path)
        number = number + 1

        if number == 200:
            break
