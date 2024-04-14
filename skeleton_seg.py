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

from utils import * 


def skeletonSegment(sitk_lung_path, bone_mask_path, lowerThreshold=129):
    ''' 完成骨骼分割，可以去除床板的影响；
    主要思路通过调节最低阈值lowerThreshold，默认为129（经验得出）
    手臂骨头有时无法分割得到，日后需要优化
    change data : 2023年6月9日

    bone 分割
    :param sitk_lung_path: 原始ct
    :param bone_mask_path 骨骼掩膜保存路径
    :param lowerThreshold 最低阈值
    '''
    print("Now start seg skeleton!")
    tme_1 = time.time()
    #sitk_src = dicomseriesReader(pathDicom)
    sitk_src = sitk.ReadImage(sitk_lung_path) # , sitk.sitkInt16
    # 1
    # sitk_seg = BinaryThreshold(sitk_src, lowervalue=100, uppervalue=3000)

    sitk_seg = sitk.BinaryThreshold(sitk_src, lowerThreshold=lowerThreshold, upperThreshold=3000, insideValue=255,
                               outsideValue=0)

    # sitk.WriteImage(sitk_seg, 'skeleton_step1.nii.gz')



    # 2  主要分割包括肺的骨头(去除毛刺)
    sitk_open = MorphologicalOperation(sitk_seg, kernelsize=2, name='open')
    # sitk.WriteImage(sitk_seg, 'skeleton_step2.nii.gz')

    # sitk.WriteImage(sitk_open, './bone_test.nii.gz')

    # 3 取最大连通域
    # sitk_open = GetLargestConnectedCompont(sitk_open) # 取最大联通域会影像骨骼的不完整分割
    # sitk.WriteImage(sitk_seg, 'skeleton_step3.nii.gz')

    array_open = sitk.GetArrayFromImage(sitk_open)
    array_seg = sitk.GetArrayFromImage(sitk_seg)

    # 4 相减
    array_mask = array_seg - array_open
    sitk_mask = sitk.GetImageFromArray(array_mask)
    sitk_mask.SetDirection(sitk_seg.GetDirection())
    sitk_mask.SetSpacing(sitk_seg.GetSpacing())
    sitk_mask.SetOrigin(sitk_seg.GetOrigin())
    # sitk.WriteImage(sitk_seg, 'skeleton_step4.nii.gz')

    # step4.最大连通域提取，去除小连接
    skeleton_mask = GetLargestConnectedCompont(sitk_mask)

    # 加上一个闭运算，减少断层
    skeleton_mask = MorphologicalOperation(skeleton_mask, kernelsize=2, name='close')

    # sitk_skeleton = GetMaskImage(sitk_src, skeleton_mask, replacevalue=-1500)

    #sitk_open = ab_GetMaskImage(sitk_open,lung_mask_path,0)
    # skeleton_mask = MorphologicalOperation(skeleton_mask, kernelsize=3, name='open')

    sitk.WriteImage(skeleton_mask, bone_mask_path)
    tme_2 = time.time()
    print('execution time: ' + str(round(tme_2-tme_1))+' s')
    print("skeleton seg over!")


