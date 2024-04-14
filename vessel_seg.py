import SimpleITK as sitk
# import matplotlib.pyplot as plt 
import numpy as np
# import nibabel as nib
import time

from skimage.filters import frangi
from utils import *
import itk



####################血管分割#############################
def window_transform(img, win_min, win_max):
    for i in range(img.shape[0]):
        img[i] = 255.0*(img[i] - win_min)/(win_max - win_min)
        min_index = img[i] < 0
        img[i][min_index] = 0
        max_index = img[i] > 255
        img[i][max_index] = 255       
        img[i] = img[i] - img[i].min()
        c = float(255)/img[i].max()
        img[i] = img[i]*c
    return img.astype(np.uint8)

def sigmoid(img, alpha, beta):
    '''
    para img：输入图像。
    para alpha：高亮血管灰度范围。
    para beta: 高亮血管中心灰度。
    '''
    img_max = img.max()
    img_min = img.min()
    return (img_max - img_min) / (1 + np.exp((beta - img) / alpha)) + img_min

def GetMaskImage(sitk_src, sitk_mask, replacevalue=0):
    """
    get mask image
    :param sitk_src:input image
    :param sitk_mask:input mask
    :param replacevalue:replacevalue of maks value equal 0
    :return:mask image
    """
    array_src = sitk.GetArrayFromImage(sitk_src)
    array_mask = sitk.GetArrayFromImage(sitk_mask)
    array_out = array_src.copy()
    array_out[array_mask == 0] = replacevalue
    outmask_sitk = sitk.GetImageFromArray(array_out)
    outmask_sitk.SetDirection(sitk_src.GetDirection())
    outmask_sitk.SetSpacing(sitk_src.GetSpacing())
    outmask_sitk.SetOrigin(sitk_src.GetOrigin())
    return outmask_sitk



# 血管分割v2
def vessel_seg_V2(input_image, output_image, lung_mask_path, pig=None):
    '''
    分割血管
    :param input_image: 输入image nii.gz
    :param output_image:血管掩膜保存路径名
    :param lung_mask_path:肺掩膜，主要用于缩小肺大小
    '''

    print("now start seg vessel!")
    tme_1 = time.time()

    print("血管开始分割")
    sigma_minimum = 2.
    sigma_maximum = 2.
    number_of_sigma_steps = 8
    lowerThreshold = 40
    #output_image = 'vessel'+str(sigma_maximum)+'.nii.gz'
    input_image = itk.imread(input_image, itk.F)
    # 1
    print("step:1")
    # tme_1 = time.time()
    ImageType = type(input_image)
    Dimension = input_image.GetImageDimension() # 3
    HessianPixelType = itk.SymmetricSecondRankTensor[itk.D, Dimension]
    HessianImageType = itk.Image[HessianPixelType, Dimension]
    objectness_filter = itk.HessianToObjectnessMeasureImageFilter[HessianImageType, ImageType].New()
    objectness_filter.SetBrightObject(True)
    objectness_filter.SetScaleObjectnessMeasure(True)
    objectness_filter.SetAlpha(0.5)
    objectness_filter.SetBeta(1.0)
    objectness_filter.SetGamma(5.0)
    multi_scale_filter = itk.MultiScaleHessianBasedMeasureImageFilter[ImageType, HessianImageType, ImageType].New()
    multi_scale_filter.SetInput(input_image)
    multi_scale_filter.SetHessianToMeasureFilter(objectness_filter)
    multi_scale_filter.SetSigmaStepMethodToLogarithmic()
    multi_scale_filter.SetSigmaMinimum(sigma_minimum)
    multi_scale_filter.SetSigmaMaximum(sigma_maximum)
    multi_scale_filter.SetNumberOfSigmaSteps(number_of_sigma_steps)
    # tme_2 = time.time()
    # print('execution time: ' + str(round(tme_2-tme_1))+' s')
    # itk.imwrite(multi_scale_filter.GetOutput(), "vessel_step1.nii.gz")
    # 2
    print("step:2")
    # tme_1 = time.time()
    OutputPixelType = itk.UC
    OutputImageType = itk.Image[OutputPixelType, Dimension]

    rescale_filter = itk.RescaleIntensityImageFilter[ImageType, OutputImageType].New()
    rescale_filter.SetInput(multi_scale_filter)
    # tme_2 = time.time()
    # print('execution time: ' + str(round(tme_2-tme_1))+' s')
    # itk.imwrite(rescale_filter.GetOutput(), "vessel_step2.nii.gz")
    # 3
    print("step:3")
    # tme_1 = time.time()
    thresholdFilter = itk.BinaryThresholdImageFilter[OutputImageType, OutputImageType].New()
    thresholdFilter.SetInput(rescale_filter.GetOutput())
    thresholdFilter.SetLowerThreshold(lowerThreshold)
    thresholdFilter.SetUpperThreshold(255)
    thresholdFilter.SetOutsideValue(0)
    thresholdFilter.SetInsideValue(255)
    # tme_2 = time.time()
    # print('execution time: ' + str(round(tme_2-tme_1))+' s')
    # itk.imwrite(thresholdFilter.GetOutput(), "vessel_step3.nii.gz")
    # 4
    print("step:4")
    # tme_1 = time.time()
    #localtime = time.asctime( time.localtime(time.time()) )
    
    itk.imwrite(thresholdFilter.GetOutput(), output_image)
    # tme_2 = time.time()
    # print('execution time: ' + str(round(tme_2-tme_1))+' s')
    print("血管分割结束")

    print("血管开始优化大小")
    #对肺mask进行腐蚀
    sitk_mask = sitk.ReadImage(lung_mask_path)
    kernelsize = 4
    new_sitk_mask = sitk.BinaryErode(sitk_mask!= 0, [kernelsize,kernelsize,kernelsize])

    '''
    vessle_mask_path:血管掩膜
    new_vessle_mask_path:优化后血管掩膜的保存路径
    '''
    vessle = sitk.ReadImage(output_image)
    vessle = MorphologicalOperation(vessle, kernelsize=2, name='close')
    new_vessle_mask = GetMaskImage(vessle, new_sitk_mask, 0)

    # 优化血管大小
    new_vessle_mask = remove_small_region(new_vessle_mask, 256)

    sitk.WriteImage(new_vessle_mask, output_image)
    tme_2 = time.time()
    print('execution time: ' + str(round(tme_2-tme_1))+' s')


