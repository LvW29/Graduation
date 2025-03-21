from __future__ import print_function, division
import numpy as np
import SimpleITK as sitk
import os
import pandas as pd

flair_name = "_flair.nii.gz"
t1_name = "_t1.nii.gz"
t1ce_name = "_t1ce.nii.gz"
t2_name = "_t2.nii.gz"
mask_name = "_seg.nii.gz"

input_bratshgg_path = r"..\unprocessed\2-MICCAI_BraTS_2018\MICCAI_BraTS_2018_Data_Training\HGG"
input_bratslgg_path = r"..\unprocessed\2-MICCAI_BraTS_2018\MICCAI_BraTS_2018_Data_Training\LGG"
input_bratshgg_path2019 = r"..\unprocessed\MICCAI_BraTS_2019_Data_Training\HGG"
input_bratslgg_path2019 = r"..\unprocessed\MICCAI_BraTS_2019_Data_Training\LGG"

def makedir(createdDir):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    abs_path = os.path.join(script_dir, createdDir)
    os.makedirs(abs_path, exist_ok=True)

bratshgg_path = input_bratshgg_path
bratslgg_path = input_bratslgg_path
if not os.path.exists(bratshgg_path):
    print("bratshgg_path 不存在")
    exit()
if not os.path.exists(bratslgg_path):
    print("bratslgg_path 不存在")
    exit()

bratshgg_path2 = input_bratshgg_path2019
bratslgg_path2 = input_bratslgg_path2019
if not os.path.exists(bratshgg_path2):
    print("bratshgg_path2019 不存在")
    exit()
if not os.path.exists(bratslgg_path2):
    print("bratslgg_path2019 不存在")
    exit()

output_trainImage = r"../processed/3D/trainImage"
output_trainMask = r"../processed/3D/trainMask"
output_testImage = r"../processed/3D/testImage"
output_testMask = r"../processed/3D/testMask"
makedir(output_trainImage)
makedir(output_trainMask)
makedir(output_testImage)
makedir(output_testMask)

trainImage = output_trainImage
trainMask = output_trainMask
testImage = output_testImage
testMask = output_testMask

def file_name_path(file_dir, dir=True, file=False):
    for root, dirs, files in os.walk(file_dir):
        if len(dirs) and dir:
            return dirs
        if len(files) and file:
            return files

def normalize(slice, bottom=99, down=1):
    b = np.percentile(slice, bottom)
    t = np.percentile(slice, down)
    slice = np.clip(slice, t, b)

    image_nonzero = slice[np.nonzero(slice)]
    if np.std(slice) == 0 or np.std(image_nonzero) == 0:
        return slice
    else:
        tmp = (slice - np.mean(image_nonzero)) / np.std(image_nonzero)
        tmp[tmp == tmp.min()] = -9
        return tmp

def crop_center(img, crop_size):
    height, width = img[0].shape
    starth = height // 2 - (crop_size // 2)
    startw = width // 2 - (crop_size // 2)
    startz = img.shape[0] // 2 - (crop_size // 2)
    return img[startz:startz + crop_size, starth:starth + crop_size, startw:startw + crop_size]

# 处理数据集的函数
def process_dataset(data_path, part, output_image_dir, output_mask_dir):
    path_list = file_name_path(data_path)
    for subsetindex in range(len(path_list)):
        print(path_list[subsetindex])
        # 1、读取数据
        brats_subset_path = data_path + "/" + str(path_list[subsetindex]) + "/"
        flair_image = brats_subset_path + str(path_list[subsetindex]) + flair_name
        t1_image = brats_subset_path + str(path_list[subsetindex]) + t1_name
        t1ce_image = brats_subset_path + str(path_list[subsetindex]) + t1ce_name
        t2_image = brats_subset_path + str(path_list[subsetindex]) + t2_name
        mask_image = brats_subset_path + str(path_list[subsetindex]) + mask_name
        flair_src = sitk.ReadImage(flair_image, sitk.sitkInt16)
        t1_src = sitk.ReadImage(t1_image, sitk.sitkInt16)
        t1ce_src = sitk.ReadImage(t1ce_image, sitk.sitkInt16)
        t2_src = sitk.ReadImage(t2_image, sitk.sitkInt16)
        mask = sitk.ReadImage(mask_image, sitk.sitkUInt8)
        flair_array = sitk.GetArrayFromImage(flair_src)
        t1_array = sitk.GetArrayFromImage(t1_src)
        t1ce_array = sitk.GetArrayFromImage(t1ce_src)
        t2_array = sitk.GetArrayFromImage(t2_src)
        mask_array = sitk.GetArrayFromImage(mask)
        # 2、人工加入切片
        myblackslice = np.zeros([240, 240])
        flair_array = np.insert(flair_array, 0, myblackslice, axis=0)
        flair_array = np.insert(flair_array, 0, myblackslice, axis=0)
        flair_array = np.insert(flair_array, 0, myblackslice, axis=0)
        flair_array = np.insert(flair_array, flair_array.shape[0], myblackslice, axis=0)
        flair_array = np.insert(flair_array, flair_array.shape[0], myblackslice, axis=0)
        t1_array = np.insert(t1_array, 0, myblackslice, axis=0)
        t1_array = np.insert(t1_array, 0, myblackslice, axis=0)
        t1_array = np.insert(t1_array, 0, myblackslice, axis=0)
        t1_array = np.insert(t1_array, t1_array.shape[0], myblackslice, axis=0)
        t1_array = np.insert(t1_array, t1_array.shape[0], myblackslice, axis=0)
        t1ce_array = np.insert(t1ce_array, 0, myblackslice, axis=0)
        t1ce_array = np.insert(t1ce_array, 0, myblackslice, axis=0)
        t1ce_array = np.insert(t1ce_array, 0, myblackslice, axis=0)
        t1ce_array = np.insert(t1ce_array, t1ce_array.shape[0], myblackslice, axis=0)
        t1ce_array = np.insert(t1ce_array, t1ce_array.shape[0], myblackslice, axis=0)
        t2_array = np.insert(t2_array, 0, myblackslice, axis=0)
        t2_array = np.insert(t2_array, 0, myblackslice, axis=0)
        t2_array = np.insert(t2_array, 0, myblackslice, axis=0)
        t2_array = np.insert(t2_array, t2_array.shape[0], myblackslice, axis=0)
        t2_array = np.insert(t2_array, t2_array.shape[0], myblackslice, axis=0)
        mask_array = np.insert(mask_array, 0, myblackslice, axis=0)
        mask_array = np.insert(mask_array, 0, myblackslice, axis=0)
        mask_array = np.insert(mask_array, 0, myblackslice, axis=0)
        mask_array = np.insert(mask_array, mask_array.shape[0], myblackslice, axis=0)
        mask_array = np.insert(mask_array, mask_array.shape[0], myblackslice, axis=0)

        # 3、对四个模态分别进行标准化
        flair_array_nor = normalize(flair_array)
        t1_array_nor = normalize(t1_array)
        t1ce_array_nor = normalize(t1ce_array)
        t2_array_nor = normalize(t2_array)

        # 4、裁剪
        crop_size = 160
        flair_crop = crop_center(flair_array_nor, crop_size)
        t1_crop = crop_center(t1_array_nor, crop_size)
        t1ce_crop = crop_center(t1ce_array_nor, crop_size)
        t2_crop = crop_center(t2_array_nor, crop_size)
        mask_crop = crop_center(mask_array, crop_size)

        # 5、合并和保存
        fourmodelimagearray = np.zeros((crop_size, crop_size, crop_size, 4), np.float64)
        filepath1 = output_image_dir + "\\" + part + "_" + path_list[subsetindex] + ".npy"
        filepath = output_mask_dir + "\\" + part + "_" + path_list[subsetindex] + ".npy"
        flairimage = flair_crop.astype(np.float64)
        fourmodelimagearray[:, :, :, 0] = flairimage
        t1image = t1_crop.astype(np.float64)
        fourmodelimagearray[:, :, :, 1] = t1image
        t1ceimage = t1ce_crop.astype(np.float64)
        fourmodelimagearray[:, :, :, 2] = t1ceimage
        t2image = t2_crop.astype(np.float64)
        fourmodelimagearray[:, :, :, 3] = t2image

        np.save(filepath1, fourmodelimagearray)
        np.save(filepath, mask_crop)

# 处理训练集 HGG
process_dataset(bratshgg_path, "hgg", trainImage, trainMask)
# 处理训练集 LGG
process_dataset(bratslgg_path, "lgg", trainImage, trainMask)
# 处理测试集 HGG（这里假设你有对应的测试集路径）
# process_dataset(input_bratshgg_test_path, "hgg", testImage, testMask)
# 处理测试集 LGG（这里假设你有对应的测试集路径）
# process_dataset(input_bratslgg_test_path, "lgg", testImage, testMask)

# 从BraTS2019多的数据集作为测试集
def compare_site_names(one, two):
    df1 = pd.read_csv(one)
    df2 = pd.read_csv(two)
    diff = set(df2['name']) - set(df1['name'])
    return list(diff)

hgg_diff = compare_site_names("18hgg.csv", "19hgg.csv")
lgg_diff = compare_site_names("18lgg.csv", "19lgg.csv")

pathhgg_list = []
pathlgg_list = []

for idx in range(len(hgg_diff)):
    mystr = "BraTS19" + hgg_diff[idx]
    pathhgg_list.append(mystr)

for idx in range(len(lgg_diff)):
    mystr = "BraTS19" + lgg_diff[idx]
    pathlgg_list.append(mystr)

def process_data(part, path_list, data_path):
    for subsetindex in range(len(path_list)):
        print(path_list[subsetindex])
        # 1、读取数据
        brats_subset_path = data_path + "/" + str(path_list[subsetindex]) + "/"
        flair_image = brats_subset_path + str(path_list[subsetindex]) + flair_name
        t1_image = brats_subset_path + str(path_list[subsetindex]) + t1_name
        t1ce_image = brats_subset_path + str(path_list[subsetindex]) + t1ce_name
        t2_image = brats_subset_path + str(path_list[subsetindex]) + t2_name
        mask_image = brats_subset_path + str(path_list[subsetindex]) + mask_name
        flair_src = sitk.ReadImage(flair_image, sitk.sitkInt16)
        t1_src = sitk.ReadImage(t1_image, sitk.sitkInt16)
        t1ce_src = sitk.ReadImage(t1ce_image, sitk.sitkInt16)
        t2_src = sitk.ReadImage(t2_image, sitk.sitkInt16)
        mask = sitk.ReadImage(mask_image, sitk.sitkUInt8)
        flair_array = sitk.GetArrayFromImage(flair_src)
        t1_array = sitk.GetArrayFromImage(t1_src)
        t1ce_array = sitk.GetArrayFromImage(t1ce_src)
        t2_array = sitk.GetArrayFromImage(t2_src)
        mask_array = sitk.GetArrayFromImage(mask)
        # 2、人工加入切片
        myblackslice = np.zeros([240, 240])
        flair_array = np.insert(flair_array, 0, myblackslice, axis=0)
        flair_array = np.insert(flair_array, 0, myblackslice, axis=0)
        flair_array = np.insert(flair_array, 0, myblackslice, axis=0)
        flair_array = np.insert(flair_array, flair_array.shape[0], myblackslice, axis=0)
        flair_array = np.insert(flair_array, flair_array.shape[0], myblackslice, axis=0)
        t1_array = np.insert(t1_array, 0, myblackslice, axis=0)
        t1_array = np.insert(t1_array, 0, myblackslice, axis=0)
        t1_array = np.insert(t1_array, 0, myblackslice, axis=0)
        t1_array = np.insert(t1_array, t1_array.shape[0], myblackslice, axis=0)
        t1_array = np.insert(t1_array, t1_array.shape[0], myblackslice, axis=0)
        t1ce_array = np.insert(t1ce_array, 0, myblackslice, axis=0)
        t1ce_array = np.insert(t1ce_array, 0, myblackslice, axis=0)
        t1ce_array = np.insert(t1ce_array, 0, myblackslice, axis=0)
        t1ce_array = np.insert(t1ce_array, t1ce_array.shape[0], myblackslice, axis=0)
        t1ce_array = np.insert(t1ce_array, t1ce_array.shape[0], myblackslice, axis=0)
        t2_array = np.insert(t2_array, 0, myblackslice, axis=0)
        t2_array = np.insert(t2_array, 0, myblackslice, axis=0)
        t2_array = np.insert(t2_array, 0, myblackslice, axis=0)
        t2_array = np.insert(t2_array, t2_array.shape[0], myblackslice, axis=0)
        t2_array = np.insert(t2_array, t2_array.shape[0], myblackslice, axis=0)
        mask_array = np.insert(mask_array, 0, myblackslice, axis=0)
        mask_array = np.insert(mask_array, 0, myblackslice, axis=0)
        mask_array = np.insert(mask_array, 0, myblackslice, axis=0)
        mask_array = np.insert(mask_array, mask_array.shape[0], myblackslice, axis=0)
        mask_array = np.insert(mask_array, mask_array.shape[0], myblackslice, axis=0)
        # 3、对四个模态分别进行标准化
        flair_array_nor = normalize(flair_array)
        t1_array_nor = normalize(t1_array)
        t1ce_array_nor = normalize(t1ce_array)
        t2_array_nor = normalize(t2_array)
        # 4、裁剪
        flair_crop = crop_ceter(flair_array_nor, 160, 160)
        t1_crop = crop_ceter(t1_array_nor, 160, 160)
        t1ce_crop = crop_ceter(t1ce_array_nor, 160, 160)
        t2_crop = crop_ceter(t2_array_nor, 160, 160)
        mask_crop = crop_ceter(mask_array, 160, 160)
        # 去掉分块处理部分

        # 5、合并和保存
        imagez, height, width = flair_crop.shape
        fourmodelimagearray = np.zeros((imagez, height, width, 4), np.float64)
        filepath1 = testImage + "\\" + part + "_" + path_list[subsetindex] + ".npy"
        filepath = testMask + "\\" + part + "_" + path_list[subsetindex] + ".npy"
        flairimage = flair_crop.astype(np.float64)
        fourmodelimagearray[:, :, :, 0] = flairimage
        t1image = t1_crop.astype(np.float64)
        fourmodelimagearray[:, :, :, 1] = t1image
        t1ceimage = t1ce_crop.astype(np.float64)
        fourmodelimagearray[:, :, :, 2] = t1ceimage
        t2image = t2_crop.astype(np.float64)
        fourmodelimagearray[:, :, :, 3] = t2image
        np.save(filepath1, fourmodelimagearray)

        wt_tc_etMaskArray = np.zeros((imagez, height, width, 3), np.uint8)
        WT_Label = mask_crop.copy()
        WT_Label[mask_crop == 1] = 1.
        WT_Label[mask_crop == 2] = 1.
        WT_Label[mask_crop == 4] = 1.
        TC_Label = mask_crop.copy()
        TC_Label[mask_crop == 1] = 1.
        TC_Label[mask_crop == 2] = 0.
        TC_Label[mask_crop == 4] = 1.
        ET_Label = mask_crop.copy()
        ET_Label[mask_crop == 1] = 0.
        ET_Label[mask_crop == 2] = 0.
        ET_Label[mask_crop == 4] = 1.
        wt_tc_etMaskArray[:, :, :, 0] = WT_Label
        wt_tc_etMaskArray[:, :, :, 1] = TC_Label
        wt_tc_etMaskArray[:, :, :, 2] = ET_Label
        np.save(filepath, wt_tc_etMaskArray)

part = "hgg"
process_data(part, pathhgg_list, bratshgg_path2)
print("HGG 数据处理完成！")
