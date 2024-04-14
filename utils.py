import SimpleITK as sitk
# import matplotlib.pyplot as plt 
import numpy as np
# import nibabel as nib
import time
import cv2
from skimage.filters import frangi
from skimage import measure
import itk
import copy
import skimage



# 去除小区域
def remove_small_region(input_img, remove_size=256):

    input_ar = sitk.GetArrayFromImage(input_img)
    output_ar = copy.deepcopy(input_ar)


    print("volarray_sum:", sum(sum(sum(input_ar))))

    label = skimage.measure.label(input_ar, connectivity=2)

    props = skimage.measure.regionprops(label)
    print(len(props))
    numPix = []
    for ia in range(len(props)):
        numPix += [props[ia].area]

    numPix_ar = np.array(numPix)
    index = np.squeeze(np.array(np.where(numPix_ar < remove_size)))

    for ind in index:
        output_ar[label==ind+1] = 0

    l = sitk.GetImageFromArray(output_ar)
    l.SetSpacing(input_img.GetSpacing())
    l.SetOrigin(input_img.GetOrigin())
    l.SetDirection(input_img.GetDirection())
    

    return l


# 窗宽窗位调整
def window_intensity(itk_image,hu_min, hu_max):
    """窗宽窗位调整 \n
    args: \n
        itk_image: simple itk 读取的图像 \n
        hu_min: 窗范围最小值 \n
        hu_max: 窗范围最大值 \n
    return: 调整窗宽窗位后的图像 \n
    """
    ww_filter = sitk.IntensityWindowingImageFilter()

    ww_filter.SetWindowMinimum(hu_min)
    ww_filter.SetWindowMaximum(hu_max)
    ww_filter.SetOutputMinimum(hu_min)
    ww_filter.SetOutputMaximum(hu_max)

    return ww_filter.Execute(itk_image)


# 得到最大连通域
def GetLargestConnectedCompont(binarysitk_image):
    """
    save largest object
    :param sitk_maskimg:binary itk image
    :return: largest region binary image
    """
    cc = sitk.ConnectedComponent(binarysitk_image)
    stats = sitk.LabelIntensityStatisticsImageFilter()
    stats.SetGlobalDefaultNumberOfThreads(8)
    stats.Execute(cc, binarysitk_image)
    maxlabel = 0
    maxsize = 0
    size_list = []
    for l in stats.GetLabels():
        size = stats.GetPhysicalSize(l)
        size_list.append(size)
        if maxsize < size:
            maxlabel = l
            maxsize = size
    labelmaskimage = sitk.GetArrayFromImage(cc) # 带label不同值的mask


    outmask = labelmaskimage.copy()
    outmask[labelmaskimage == maxlabel] = 1
    outmask[labelmaskimage != maxlabel] = 0


    outmask_sitk = sitk.GetImageFromArray(outmask)
    outmask_sitk.SetDirection(binarysitk_image.GetDirection())
    outmask_sitk.SetSpacing(binarysitk_image.GetSpacing())
    outmask_sitk.SetOrigin(binarysitk_image.GetOrigin())
    return outmask_sitk

# 得到最大的两个连通域，用于肺部掩膜的处理
def Get2LargestConnectedCompont(binarysitk_image):
    """
    save the 2nd largest object
    :param sitk_maskimg:binary itk image
    :return: largest region binary image
    """
    cc = sitk.ConnectedComponent(binarysitk_image)
    stats = sitk.LabelIntensityStatisticsImageFilter()
    stats.SetGlobalDefaultNumberOfThreads(8)
    stats.Execute(cc, binarysitk_image)
    maxlabel = 0
    maxsize = 0
    old_maxlabel = None
    size_list = []
    for l in stats.GetLabels():
        size = stats.GetPhysicalSize(l)
        size_list.append(size)
        if maxsize < size:
            old_maxlabel = maxlabel
            old_maxsize = maxsize
            maxlabel = l
            maxsize = size
    labelmaskimage = sitk.GetArrayFromImage(cc) # 带label不同值的mask


    outmask = labelmaskimage.copy()
    outmask[labelmaskimage == maxlabel] = 1
    outmask[labelmaskimage != maxlabel] = 0

    if old_maxlabel != None:
        outmask[labelmaskimage == old_maxlabel] = 1


    outmask_sitk = sitk.GetImageFromArray(outmask)
    outmask_sitk.SetDirection(binarysitk_image.GetDirection())
    outmask_sitk.SetSpacing(binarysitk_image.GetSpacing())
    outmask_sitk.SetOrigin(binarysitk_image.GetOrigin())
    return outmask_sitk

def MorphologicalOperation(sitk_maskimg, kernelsize, name='open'):
    """
    morphological operation
    :param sitk_maskimg:input binary image
    :param kernelsize:kernel size
    :param name:operation name
    :return:binary image
    """
    if name == 'open':
        morphoimage = sitk.BinaryMorphologicalOpening(sitk_maskimg != 0)
        return morphoimage
    if name == 'close':
        morphoimage = sitk.BinaryMorphologicalClosing(sitk_maskimg != 0)
        return morphoimage
    if name == 'dilate':
        morphoimage = sitk.BinaryDilate(sitk_maskimg != 0, kernelsize)
        return morphoimage
    if name == 'erode':
        morphoimage = sitk.BinaryErode(sitk_maskimg != 0, kernelsize)
        return morphoimage

