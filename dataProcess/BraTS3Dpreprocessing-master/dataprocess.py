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

input_bratshgg_path = r"..\..\..\data\unprocessed\2-MICCAI_BraTS_2018\MICCAI_BraTS_2018_Data_Training\HGG"
input_bratslgg_path = r"..\..\..\data\unprocessed\2-MICCAI_BraTS_2018\MICCAI_BraTS_2018_Data_Training\LGG"
input_bratshgg_path2019 = r"..\..\..\data\unprocessed\MICCAI_BraTS_2019_Data_Training\HGG"
input_bratslgg_path2019 = r"..\..\..\data\unprocessed\MICCAI_BraTS_2019_Data_Training\LGG"

BLOCKSIZE = (160, 160, 160)  # 每个分块的大小


def makedir(createdDir):
    script_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前脚本的绝对路径
    abs_path = os.path.join(script_dir, createdDir)  # 将相对路径转换为绝对路径
    os.makedirs(abs_path, exist_ok=True)  # 创建文件夹


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

output_trainImage = r"../processed/3D-noblock/trainImage"
output_trainMask = r"../processed/3D-noblock/trainMask"
output_testImage = r"../processed/3D-noblock/testImage"
output_testMask = r"../processed/3D-noblock/testMask"
makedir(output_trainImage)
makedir(output_trainMask)
makedir(output_testImage)
makedir(output_testMask)

trainImage = output_trainImage
trainMask = output_trainMask
# if not os.path.exists(trainImage):
#     os.makedirs(output_trainImage)
#     print("trainImage 输出目录创建成功")
# if not os.path.exists(trainMask):
#     os.makedirs(trainMask)
#     print("trainMask 输出目录创建成功")

testImage = output_testImage
testMask = output_testMask


# if not os.path.exists(testImage):
#     os.makedirs(testImage)
#     print("testImage 输出目录创建成功")
# if not os.path.exists(testMask):
#     os.makedirs(testMask)
#     print("testMask 输出目录创建成功")

def file_name_path(file_dir, dir=True, file=False):
    """
    get root path,sub_dirs,all_sub_files
    :param file_dir:
    :return: dir or file
    """
    for root, dirs, files in os.walk(file_dir):
        if len(dirs) and dir:
            print("sub_dirs:", dirs)
            return dirs
        if len(files) and file:
            print("files:", files)
            return files


pathhgg_list = file_name_path(bratshgg_path)
pathlgg_list = file_name_path(bratslgg_path)


def normalize(slice, bottom=99, down=1):
    """
    normalize image with mean and std for regionnonzero,and clip the value into range
    :param slice:
    :param bottom:
    :param down:
    :return:
    """
    b = np.percentile(slice, bottom)
    t = np.percentile(slice, down)
    slice = np.clip(slice, t, b)

    image_nonzero = slice[np.nonzero(slice)]
    if np.std(slice) == 0 or np.std(image_nonzero) == 0:
        return slice
    else:
        tmp = (slice - np.mean(image_nonzero)) / np.std(image_nonzero)
        # since the range of intensities is between 0 and 5000 ,
        # the min in the normalized slice corresponds to 0 intensity in unnormalized slice
        # the min is replaced with -9 just to keep track of 0 intensities
        # so that we can discard those intensities afterwards when sampling random patches
        tmp[tmp == tmp.min()] = -9
        return tmp


def crop_ceter(img, croph, cropw):
    # for n_slice in range(img.shape[0]):
    height, width = img[0].shape
    starth = height // 2 - (croph // 2)
    startw = width // 2 - (cropw // 2)
    return img[:, starth:starth + croph, startw:startw + cropw]


def process_brats_data(brats_path, part, output_image, output_mask):
    path_list = file_name_path(brats_path)
    for subsetindex in range(len(path_list)):
        print(path_list[subsetindex])
        # 1、读取数据
        brats_subset_path = brats_path + "/" + str(path_list[subsetindex]) + "/"
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
        arrays = [flair_array, t1_array, t1ce_array, t2_array, mask_array]
        for i in range(len(arrays)):
            arr = arrays[i]
            # 插入3片头部黑切片
            arr = np.insert(arr, 0, myblackslice, axis=0)
            arr = np.insert(arr, 0, myblackslice, axis=0)
            arr = np.insert(arr, 0, myblackslice, axis=0)
            # 插入2片尾部黑切片
            arr = np.insert(arr, arr.shape[0], myblackslice, axis=0)
            arr = np.insert(arr, arr.shape[0], myblackslice, axis=0)
            arrays[i] = arr  # 保存修改后的数组

        # 重新分配数组
        flair_array, t1_array, t1ce_array, t2_array, mask_array = arrays
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
        # 5、合并和保存
        save_patches(flair_crop, t1_crop, t1ce_crop, t2_crop, mask_crop, part, path_list[subsetindex], output_image, output_mask)

def save_patches(flair_crop, t1_crop, t1ce_crop, t2_crop, mask_crop, part, subsetindex, output_image, output_mask):
    samples_flair = np.array(flair_crop).reshape((1, BLOCKSIZE[0], BLOCKSIZE[1], BLOCKSIZE[2]))
    samples_t1 = np.array(t1_crop).reshape((1, BLOCKSIZE[0], BLOCKSIZE[1], BLOCKSIZE[2]))
    samples_t1ce = np.array(t1ce_crop).reshape((1, BLOCKSIZE[0], BLOCKSIZE[1], BLOCKSIZE[2]))
    samples_t2 = np.array(t2_crop).reshape((1, BLOCKSIZE[0], BLOCKSIZE[1], BLOCKSIZE[2]))
    mask_samples = np.array(mask_crop).reshape((1, BLOCKSIZE[0], BLOCKSIZE[1], BLOCKSIZE[2]))
    samples, imagez, height, width = np.shape(samples_flair)[0], np.shape(samples_flair)[1], \
        np.shape(samples_flair)[2], np.shape(samples_flair)[3]
    for j in range(samples):
        """
        merage 4 model image into 4 channel (imagez,width,height,channel)
        """
        fourmodelimagearray = np.zeros((imagez, height, width, 4), np.float64)
        filepath1 = output_image + "\\" + part + "_" + subsetindex + "_" + str(j) + ".npy"
        filepath = output_mask + "\\" + part + "_" + subsetindex + "_" + str(j) + ".npy"
        flairimage = samples_flair[j, :, :, :]
        flairimage = flairimage.astype(np.float64)
        fourmodelimagearray[:, :, :, 0] = flairimage
        t1image = samples_t1[j, :, :, :]
        t1image = t1image.astype(np.float64)
        fourmodelimagearray[:, :, :, 1] = t1image
        t1ceimage = samples_t1ce[j, :, :, :]
        t1ceimage = t1ceimage.astype(np.float64)
        fourmodelimagearray[:, :, :, 2] = t1ceimage
        t2image = samples_t2[j, :, :, :]
        t2image = t2image.astype(np.float64)
        fourmodelimagearray[:, :, :, 3] = t2image
        np.save(filepath1, fourmodelimagearray)

        wt_tc_etMaskArray = np.zeros((imagez, height, width, 3), np.uint8)
        mask_one_sample = mask_samples[j, :, :, :]
        WT_Label = mask_one_sample.copy()
        WT_Label[mask_one_sample == 1] = 1.
        WT_Label[mask_one_sample == 2] = 1.
        WT_Label[mask_one_sample == 4] = 1.
        TC_Label = mask_one_sample.copy()
        TC_Label[mask_one_sample == 1] = 1.
        TC_Label[mask_one_sample == 2] = 0.
        TC_Label[mask_one_sample == 4] = 1.
        ET_Label = mask_one_sample.copy()
        ET_Label[mask_one_sample == 1] = 0.
        ET_Label[mask_one_sample == 2] = 0.
        ET_Label[mask_one_sample == 4] = 1.
        wt_tc_etMaskArray[:, :, :, 0] = WT_Label
        wt_tc_etMaskArray[:, :, :, 1] = TC_Label
        wt_tc_etMaskArray[:, :, :, 2] = ET_Label
        np.save(filepath, wt_tc_etMaskArray)


def compare_site_names(one, two):
    df1 = pd.read_csv(one, header=None)
    df2 = pd.read_csv(two, header=None)
    data1 = df1.to_dict(orient='list')[0]
    data2 = df2.to_dict(orient='list')[0]
    diff1 = set(data1).difference(data2)
    diff2 = set(data2).difference(data1)
    return list(diff2)


# 处理BraTS2018训练集
process_brats_data(bratshgg_path, "hgg", output_trainImage, output_trainMask)
process_brats_data(bratslgg_path, "lgg", output_trainImage, output_trainMask)

# 从BraTS2019多的数据集作为测试集
hgg_diff = compare_site_names("18hgg.csv", "19hgg.csv")
lgg_diff = compare_site_names("18lgg.csv", "19lgg.csv")

pathhgg_list = ["BraTS19" + idx for idx in hgg_diff]
pathlgg_list = ["BraTS19" + idx for idx in lgg_diff]

# 处理BraTS2019测试集
process_brats_data(bratshgg_path2, "hgg", output_testImage, output_testMask)
process_brats_data(bratslgg_path2, "lgg", output_testImage, output_testMask)

print("Done!")
