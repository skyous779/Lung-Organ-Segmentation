import SimpleITK as sitk

import time


############################### 气管与肺分割 #########################

def skin_window_transform(ct_array, windowWidth, windowCenter, normal=False):
    """
    return: trucated image according to window center and window width
    and normalized to [0,1]
    """
    minWindow = float(windowCenter) - 0.5*float(windowWidth)
    newimg = (ct_array - minWindow) / float(windowWidth)
    newimg[newimg < 0] = 0
    newimg[newimg > 1] = 1
    if not normal:
        newimg = (newimg * 255).astype('float32')
    return newimg

def skin_mask(nii_path, lung_mask_path, skin_save_path):
    '''
    param nii_path: 输入image nii.gz
    param lung_mask_path: 肺与气管粗分割掩膜
    param skin_save_path: 皮肤保存路径
    尝试使用开运算 没能去掉一些杂点，后期优化思路，开运算加取最大连通域
    '''
    print("now start seg  skin!")
    tme_1 = time.time()

    image = sitk.ReadImage(nii_path, sitk.sitkFloat32)

    origin = image.GetOrigin()
    spacing = image.GetSpacing()
    direction = image.GetDirection()

    ## 肺窗/二值处理
    array = sitk.GetArrayFromImage(image)
    array[array > 0] = 255
    array=skin_window_transform(array,400,-200,False)
    array[array > 0] = 255
    image1=sitk.GetImageFromArray(array)

    image1.SetDirection(direction)
    image1.SetSpacing(spacing)
    image1.SetOrigin(origin)

    # sitk.WriteImage(image1,"skin1.nii.gz")

    ### 提取最大连通量
    image1 = sitk.Cast(image1, sitk.sitkInt16)
    cc = sitk.ConnectedComponent(image1)  #只支持16位
    stats = sitk.LabelIntensityStatisticsImageFilter()
    stats.SetGlobalDefaultNumberOfThreads(8)
    stats.Execute(cc, image1)
    maxlabel = 0
    maxsize = 0
    for l in stats.GetLabels():
        size = stats.GetPhysicalSize(l)
        if maxsize < size:
            maxlabel = l
            maxsize = size
    labelmaskimage = sitk.GetArrayFromImage(cc)
    outmask = labelmaskimage.copy()
    outmask[labelmaskimage == maxlabel] = 1
    outmask[labelmaskimage != maxlabel] = 0

    outmasksitk = sitk.GetImageFromArray(outmask)

    outmasksitk.SetDirection(direction)
    outmasksitk.SetSpacing(spacing)
    outmasksitk.SetOrigin(origin)

    # sitk.WriteImage(outmasksitk,"skin2.nii.gz")

    ### 开运算，去掉小白点
    kernelsize = (5,5,5)
    image1 = sitk.BinaryMorphologicalOpening(outmasksitk != 0, kernelsize)

    image1.SetDirection(direction)
    image1.SetSpacing(spacing)
    image1.SetOrigin(origin)

    # sitk.WriteImage(image1,"out3.nii.gz")

    # 使用肺部掩膜和闭运算，填充孔洞
    # 使用到粗分割肺的掩膜
    lung_mask_sitk = sitk.ReadImage(lung_mask_path)

    #对肺掩膜做一个闭运算,然后填充
    lung_mask_sitk_1 = sitk.BinaryDilate(lung_mask_sitk != 0, (5,5,5))
    mask = sitk.GetArrayFromImage(lung_mask_sitk_1)

    image4_array = sitk.GetArrayFromImage(image1)
    image4_array[mask == 1] = 1


    image1 = sitk.GetImageFromArray(image4_array)

    image1 = sitk.BinaryDilate(image1 != 0, (5,5,5))


    image1.SetOrigin(origin)
    image1.SetSpacing(spacing)
    image1.SetDirection(direction)

    sitk.WriteImage(image1, skin_save_path)

    tme_2 = time.time()
    print('execution time: ' + str(round(tme_2-tme_1))+' s')
    
    ## sobel 算子提取边界


    # change data type before edge detection
    image1 = sitk.Cast(image1, sitk.sitkFloat32)

    sobel_op = sitk.SobelEdgeDetectionImageFilter()
    image1 = sobel_op.Execute(image1)
    image1 = sitk.Cast(image1, sitk.sitkInt16)

    # 二值化皮肤mask
    sobel_array = sitk.GetArrayFromImage(image1)
    outmask = sobel_array.copy()
    outmask[sobel_array != 0] = 1

    image1 = sitk.GetImageFromArray(outmask)

    image1.SetOrigin(origin)
    image1.SetSpacing(spacing)
    image1.SetDirection(direction)
    sitk.WriteImage(image1, skin_save_path)
