from datasets import load_dataset
from PIL import Image
import numpy as np
import glob
import os
import cv2
import matplotlib.pyplot as plt


# Image transformer
def transformer(image_path, markup_path, type, save_path):
    image_gray1 = Image.open(image_path).convert("L")
    image_gray2 = Image.open(markup_path).convert("L")

    if image_gray1.size != image_gray2.size:
        image_gray2 = image_gray2.resize(image_gray1.size)

    array_gray1 = np.array(image_gray1)
    array_gray2 = np.array(image_gray1)
    array_gray3 = np.array(image_gray2)
    array_rgb = np.stack((array_gray1, array_gray2, array_gray3), axis=2)
    image_rgb = Image.fromarray(array_rgb)

    image_name = os.path.basename(image_path)
    image_rgb.save(f"{save_path}/{type}/{image_name}")


# Create dataset
def dataset_creator(images_dir_test, markup_dir_test, images_dir_train, markup_dir_train, save_dir):
    file_pattern = "*.png"
    image_paths_test = glob.glob(f"{images_dir_test}/{file_pattern}")
    for i, image_path in enumerate(image_paths_test):
        transformer(image_path,
                    os.path.join(markup_dir_test, str(i)) + ".png",
                    'training',
                    save_dir)

    image_paths_train = glob.glob(f"{images_dir_train}/{file_pattern}")

    for i, image_path in enumerate(image_paths_train):
        transformer(image_path,
                    os.path.join(markup_dir_train, str(i)) + ".png",
                    'testing',
                    save_dir)


# Crop dataset
def dataset_cropper(dataset_path, output_path, size):
    selection_cropper(f"{dataset_path}/training/",
                      f"{output_path}/training/",
                      'training',
                      size)

    selection_cropper(f"{dataset_path}/testing/",
                      f"{output_path}/testing/",
                      'testing',
                      size)


# Crop selection
def selection_cropper(image_path, output_path, type, size):
    file_pattern = "*.png"
    image_paths = glob.glob(f"{image_path}/{file_pattern}")
    num_horizontal_splits = 4
    num_vertical_splits = 3

    cnt = 0
    for i, image_path in enumerate(image_paths):
        image = Image.open(image_path)
        width, height = image.size
        split_width = width // num_horizontal_splits
        split_height = height // num_vertical_splits

        for row in range(num_vertical_splits):
            for col in range(num_horizontal_splits):
                left = col * split_width
                top = row * split_height
                right = left + split_width
                bottom = top + split_height
                split_image = image.crop((left, top, right, bottom))
                split_image.save(f"{output_path}/{type}{cnt:04d}.png")
                cnt += 1
                if cnt == size:
                    return


# Copy dataset
def dataset_copier(train_dir, test_dir, save_dir, path):
    file_pattern = "*.png"
    train_paths = glob.glob(f"{train_dir}/{file_pattern}")
    test_paths = glob.glob(f"{test_dir}/{file_pattern}")
    for i, image_path in enumerate(train_paths):
        image = Image.open(image_path)
        image.save(f"{save_dir}/training/training{str(i).zfill(4)}.png")
    for i, image_path in enumerate(test_paths):
        image = Image.open(image_path)
        image.save(f"{path}/testing{str(i).zfill(4)}.png")


# Push dataset to repo hub
def push_to_hub(data, name):
    dataset = load_dataset("imagefolder", data_dir=data)
    dataset.push_to_hub(name)


# Create result images
def result_creator(input_image, save_dir, num):
    image = Image.open(input_image)
    channels = image.split()
    gray_image_1 = channels[0].convert("L")
    gray_image_2 = channels[1].convert("L")
    gray_image_3 = channels[2].convert("L")
    gray_image_1.save(f"{save_dir}/img_{num:04d}.png")
    gray_image_2.save(f"{save_dir}/channel_2.png")
    gray_image_3.save(f"{save_dir}/img_{num:04d}_mask.png")


def histogram_comparator(sample_img, result_img_path):
    correlation_list = []
    image1 = cv2.imread(sample_img, cv2.IMREAD_GRAYSCALE)

    compared = sorted(glob.glob(f"{result_img_path}/*.png"))
    for i, current in enumerate(compared):
        if 'mask' in current:
            continue
        image2 = cv2.imread(current, cv2.IMREAD_GRAYSCALE)
        histogram1, _ = np.histogram(image1.ravel(), bins=256, range=[0, 256])
        histogram2, _ = np.histogram(image2.ravel(), bins=256, range=[0, 256])
        correlation_list.append(np.corrcoef(histogram1, histogram2)[0, 1])

    return correlation_list


def hellinger_comparator(sample_img, result_img_path):
    hellinger_list = []
    image1 = cv2.imread(sample_img, cv2.IMREAD_GRAYSCALE)

    compared = sorted(glob.glob(f"{result_img_path}/*.png"))
    for i, current in enumerate(compared):
        if 'mask' in current:
            continue
        image2 = cv2.imread(current, cv2.IMREAD_GRAYSCALE)
        histogram1, _ = np.histogram(image1.ravel(), bins=256, range=[0, 256])
        histogram2, _ = np.histogram(image2.ravel(), bins=256, range=[0, 256])
        histogram1 = histogram1 / np.sum(histogram1)
        histogram2 = histogram2 / np.sum(histogram2)
        sqrt_hist1 = np.sqrt(histogram1)
        sqrt_hist2 = np.sqrt(histogram2)
        sum_squares = np.sum((sqrt_hist1 - sqrt_hist2) ** 2)
        distance = np.sqrt(0.5 * sum_squares)
        hellinger_list.append(distance)

    return hellinger_list


def tresholding(result_img_path, save_dir):
    images_all = sorted(glob.glob(f"{result_img_path}/*.png"))
    masks = [i for i in images_all if 'mask' in i]
    images = [i for i in images_all if 'mask' not in i]
    for num, zip_image in enumerate(zip(images, masks)):
        image_path = zip_image[0]
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(f"{save_dir}/img_{num:04d}.png", image)
        mask_path = zip_image[1]
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        thresh_value = 127
        max_value = 255
        thresh_type = cv2.THRESH_BINARY
        ret, thresh_img = cv2.threshold(mask, thresh_value, max_value, thresh_type)
        cv2.imwrite(f"{save_dir}/img_{num:04d}_mask.png", thresh_img)


def combine_mask_and_image(images_path, save_dir):
    images_all = sorted(glob.glob(f"{images_path}/*.png"))
    masks = [i for i in images_all if 'mask' in i]
    images = [i for i in images_all if 'mask' not in i]
    for num, zip_image in enumerate(zip(images, masks)):
        image_path = zip_image[0]
        mask_path = zip_image[1]
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        result = np.zeros_like(image)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if mask[i, j] == 0:
                    result[i, j] = 255
                else:
                    result[i, j] = image[i, j]
        cv2.imwrite(f"{save_dir}/img_{num:04d}_only_mitochondria.png", result)


def print_statistics(method, i, data):
    print(f"{method}, original - {i}")
    print(max(data))
    print(np.argmax(data))
    print(min(data))
    print(sum(data)/len(data))
    print(np.median(data))


def compute_statistics(arrays):
    max_of_max = np.max([np.max(array) for array in arrays])
    max_of_max_index = np.argmax([np.max(array) for array in arrays])

    min_of_max = np.min([np.max(array) for array in arrays])
    min_of_max_index = np.argmin([np.max(array) for array in arrays])

    min_of_min = np.min([np.min(array) for array in arrays])
    min_of_min_index = np.argmin([np.min(array) for array in arrays])

    max_of_median = np.max([np.median(array) for array in arrays])
    max_of_median_index = np.argmax([np.median(array) for array in arrays])

    max_of_mean = np.max([np.mean(array) for array in arrays])
    max_of_mean_index = np.argmax([np.mean(array) for array in arrays])

    min_of_median = np.min([np.median(array) for array in arrays])
    min_of_median_index = np.argmin([np.median(array) for array in arrays])

    min_of_mean = np.min([np.mean(array) for array in arrays])
    min_of_mean_index = np.argmin([np.mean(array) for array in arrays])

    median_of_medians = np.median([np.median(array) for array in arrays])
    mean_of_means = np.mean([np.mean(array) for array in arrays])

    return {
        'max_of_max': max_of_max,
        'max_of_max_index': max_of_max_index,
        'min_of_max': min_of_max,
        'min_of_max_index': min_of_max_index,
        'min_of_min': min_of_min,
        'min_of_min_index': min_of_min_index,
        'max_of_median': max_of_median,
        'max_of_median_index': max_of_median_index,
        'max_of_mean': max_of_mean,
        'max_of_mean_index': max_of_mean_index,
        'min_of_median': min_of_median,
        'min_of_median_index': min_of_median_index,
        'min_of_mean': min_of_mean,
        'min_of_mean_index': min_of_mean_index,
        'median_of_medians': median_of_medians,
        'mean_of_means': mean_of_means
    }


def save_image_histogram(image_path, output_path, title):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
    histogram /= histogram.sum()
    plt.plot(histogram, color='black')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.savefig(output_path)
    plt.close()


def tresholding_single(img_path, save_dir):
    mask = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    thresh_value = 240
    max_value = 255
    thresh_type = cv2.THRESH_BINARY
    ret, thresh_img = cv2.threshold(mask, thresh_value, max_value, thresh_type)
    cv2.imwrite(f"{save_dir}/img_mask.png", thresh_img)
