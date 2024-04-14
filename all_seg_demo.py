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



####################气管分割#############################

def regionGrowing(grayImg, seed, threshold):
    """
    :param grayImg: 灰度图像
    :param seed: 生长起始点的位置
    :param threshold: 阈值
    :return: 取值为{0, 255}的二值图像
    """
    [maxX, maxY,maxZ] = grayImg.shape[0:3]

    # 用于保存生长点的队列
    pointQueue = []
    pointQueue.append((seed[0], seed[1],seed[2])) # 保存种子点
    outImg = np.zeros_like(grayImg)
    outImg[seed[0], seed[1],seed[2]] = 1

    pointsNum = 1
    pointsMean = float(grayImg[seed[0], seed[1],seed[2]])

    # 用于计算生长点周围26个点的位置
    Next26 = [[-1, -1, -1],[-1, 0, -1],[-1, 1, -1],
                [-1, 1, 0], [-1, -1, 0], [-1, -1, 1],
                [-1, 0, 1], [-1, 0, 0],[-1, 0, -1],
                [0, -1, -1], [0, 0, -1], [0, 1, -1],
                [0, 1, 0],[-1, 0, -1],
                [0, -1, 0],[0, -1, 1],[-1, 0, -1],
                [0, 0, 1],[1, 1, 1],[1, 1, -1],
                [1, 1, 0],[1, 0, 1],[1, 0, -1],
                [1, -1, 0],[1, 0, 0],[1, -1, -1]]

    while(len(pointQueue)>0):
        # 取出队首并删除
        growSeed = pointQueue[0]
        del pointQueue[0]

        for differ in Next26:
            growPointx = growSeed[0] + differ[0]
            growPointy = growSeed[1] + differ[1]
            growPointz = growSeed[2] + differ[2]

            # 是否是边缘点
            if((growPointx < 0) or (growPointx > maxX - 1) or
               (growPointy < 0) or (growPointy > maxY - 1) or (growPointz < 0) or (growPointz > maxZ - 1)) :
                continue

            # 是否已经被生长
            if(outImg[growPointx,growPointy,growPointz] == 1):
                continue

            data = grayImg[growPointx,growPointy,growPointz] # 该点的像素值
            # 判断条件
            # 符合条件则生长，并且加入到生长点队列中
            if(abs(data - pointsMean)<threshold):
                pointsNum += 1
                pointsMean = (pointsMean * (pointsNum - 1) + data) / pointsNum
                outImg[growPointx, growPointy,growPointz] = 1
                pointQueue.append([growPointx, growPointy,growPointz])
            
            ## 控制点数
            # if pointsNum > 70000:
            #     print(pointsMean)
            #     break

    return outImg, pointsNum 

def regionGrowing_v2(grayImg, seed, open_grow_mode = None):
    """
    寻找最佳阈值
    :param grayImg: 灰度图像
    :param seed: 生长起始点的位置
    :param threshold: 阈值
    :param open_grow_mode: 防止一些气管分割提前停止生长，导致效果不好，可以说是放开生长模式
    :return: 取值为{0, 255}的二值图像
    """

    if open_grow_mode is not None:
        pointsnum_min = 30000
    else:
        pointsnum_min = 15000
    pointsnum_min = 20000

    threshold = 0
    Lt = 0
    old_point_num = 1
    seg_start = 0 #第一次生成模式，一般都会有超过0.9以上的增加率
    while(True): # 阈值大循环
        [maxX, maxY,maxZ] = grayImg.shape[0:3]

        # 用于保存生长点的队列
        pointQueue = []
        pointQueue.append((seed[0], seed[1],seed[2])) # 保存种子点
        outImg = np.zeros_like(grayImg)
        outImg[seed[0], seed[1],seed[2]] = 1

        pointsNum = 1
        pointsMean = float(grayImg[seed[0], seed[1],seed[2]])

        # 用于计算生长点周围26个点的位置
        Next26 = [[-1, -1, -1],[-1, 0, -1],[-1, 1, -1],
                    [-1, 1, 0], [-1, -1, 0], [-1, -1, 1],
                    [-1, 0, 1], [-1, 0, 0],[-1, 0, -1],
                    [0, -1, -1], [0, 0, -1], [0, 1, -1],
                    [0, 1, 0],[-1, 0, -1],
                    [0, -1, 0],[0, -1, 1],[-1, 0, -1],
                    [0, 0, 1],[1, 1, 1],[1, 1, -1],
                    [1, 1, 0],[1, 0, 1],[1, 0, -1],
                    [1, -1, 0],[1, 0, 0],[1, -1, -1]]

        # 给定阈值后一个生长循环
        while(len(pointQueue)>0):
            # 取出队首并删除
            growSeed = pointQueue[0]
            del pointQueue[0]

            for differ in Next26:
                growPointx = growSeed[0] + differ[0]
                growPointy = growSeed[1] + differ[1]
                growPointz = growSeed[2] + differ[2]

                # 是否是边缘点
                if((growPointx < 0) or (growPointx > maxX - 1) or
                (growPointy < 0) or (growPointy > maxY - 1) or (growPointz < 0) or (growPointz > maxZ - 1)) :
                    continue

                # 是否已经被生长
                if(outImg[growPointx,growPointy,growPointz] == 1):
                    continue

                data = grayImg[growPointx,growPointy,growPointz] # 该点的像素值
                # 判断条件
                # 符合条件则生长，并且加入到生长点队列中
                if(abs(data - pointsMean)<threshold):
                    pointsNum += 1
                    pointsMean = (pointsMean * (pointsNum - 1) + data) / pointsNum
                    outImg[growPointx, growPointy,growPointz] = 1
                    pointQueue.append([growPointx, growPointy,growPointz])
                if pointsNum > 150000:
                    break

        add_pointsNum = pointsNum - old_point_num
        Lt = add_pointsNum / pointsNum


        print("threshold:", threshold, "Lt:", Lt, 'pointsNum:', pointsNum)
        if (Lt > 0.1  and seg_start == 1 and old_point_num > pointsnum_min ) or pointsNum > 100000:
            # pointsNum > 150000 气管一般不会超过这么大的像素个数
            # 150000可以说是气管的上限。
            # 27000为气管的下限
            # print("the best threshold is :", threshold)
            break
        else:
            threshold += 1
            old_point_num = pointsNum

            # 避免第一次有效生长导致停止
            if Lt > 0.9 :
                seg_start = 1

##################debug mode####################
        # if (Lt > 0.1  and seg_start == 1 and pointsNum > pointsnum_min ) or pointsNum > 150000:
        #     print("may stop") # 调试
        # threshold += 1
        # old_point_num = pointsNum
        # if Lt > 0.9 :
        #     seg_start = 1



        # threshold += 1
        # old_point_num = pointsNum

    # 舍弃
    # if Lt > 0.9 or pointsNum > 150000: # 第二次Lt高达0.9， threshold应该取前一个
    #     output_threshold = threshold - 1
    # else:
    #     output_threshold = threshold

    return threshold - 1

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


def airways_seg(image_path = "./dataset/LIDC-IDRI-0005.nii.gz",
                output_path = "'./output/airways5_filter.nii.gz'",
                seed = (236, 201, 252), threshold = 113, open_grow_mode = None):
    '''
    使用一个定点
    还有阈值
    '''

    start = time.time()

    lung_image = sitk.ReadImage(image_path)
    lung_arr = sitk.GetArrayFromImage(lung_image) # zyx
    lung_arr_T = lung_arr.transpose((2, 1, 0)) # zyx => xyz


    # threshold = 41 # 临近像素的阈值，微调可以减少误分割，和少分割 阈值越小，mask越小，默认50，需要动态设置阈值才好
    threshold = regionGrowing_v2(lung_arr_T, seed, open_grow_mode = open_grow_mode)

    trachea_img_T, pointsNum = regionGrowing(lung_arr_T, seed, threshold=threshold)
    trachea_img = trachea_img_T.transpose((2, 1, 0)) # xyz => zyx

    print('the number of pointsNum: ', pointsNum)

    pred = sitk.GetImageFromArray(trachea_img)
    pred.SetDirection(lung_image.GetDirection())
    pred.SetOrigin(lung_image.GetOrigin())
    pred.SetSpacing(lung_image.GetSpacing())

    # 闭运算填充与连接
    sitk_airways = sitk.BinaryMorphologicalClosing(pred != 0, [5,5,5])

    #取最大连通阈值 pass
    # pred = GetLargestConnectedCompont(pred)

    #填充hole
    # sitk_airways = sitk.BinaryFillhole(pred)
    
    output_path_threshold = output_path.split('.nii.gz')[0]+'_'+str(threshold)+'.nii.gz'
    print(output_path_threshold)
    sitk.WriteImage(sitk_airways, output_path_threshold)

    end = time.time()
    print('airways seg process end','time:',str(end-start))

    return sitk_airways


####################肺部分割#############################

# v1 肺实质不是实心，已去
def lung_cu_seg(img_path='./dataset/LIDC-IDRI-0005.nii.gz',
                 save_path=None,
                 seed_pts=None):
    '''
    肺部粗分割代码, 返回粗分割后的image
    params img_path: 输入肺部CT的nii.gz格式
    params save_path: 肺掩膜保存路径
    params seed_pts: 肺部两个种子点
    '''
    start = time.time()
    WINDOW_LEVEL = (1000,-500)

    # img_path = './dataset/LIDC-IDRI-0005.nii.gz'
    img = sitk.ReadImage(img_path)
    if seed_pts == None:
        seed_pts = [(150, 200, 68), (390, 255, 68)]
        print('使用默认种子点: seed_pts = [(150, 200, 68), (390, 255, 68)]')
    img_grow_Confidence = sitk.ConfidenceConnected(img, seedList=seed_pts,
                                                            numberOfIterations=0,
                                                            multiplier=2,
                                                            initialNeighborhoodRadius=1,
                                                            replaceValue=1)

    BMC = sitk.BinaryMorphologicalClosingImageFilter()
    BMC.SetKernelType(sitk.sitkBall)
    BMC.SetKernelRadius(2)
    BMC.SetForegroundValue(1)
    OUT = BMC.Execute(img_grow_Confidence)

    if save_path != None:
        sitk.WriteImage(OUT, save_path) # 保存图像

    end = time.time()
    print('lung_cu_seg process end','time:',str(end-start))

    return OUT

# v2
def lung_cu_seg_v2(img_path, remove_block=None):
    '''
    img_path : 原始肺部CT路径
    remove_block : 用于是否去除床板，如果为1，则取第二大连通域
    '''

    start = time.time()
    lung_img = sitk.ReadImage(img_path)
    #获取体数据的尺寸
    size = sitk.Image(lung_img).GetSize()
    #获取体数据direction
    direction = sitk.Image(lung_img).GetDirection()
    #获取体数据的空间尺寸
    spacing = sitk.Image(lung_img).GetSpacing()
    #获得体数据的oringin
    oringin = sitk.Image(lung_img).GetOrigin()
    #将体数据转为numpy数组
    volarray = sitk.GetArrayFromImage(lung_img)

    #根据CT值，将数据二值化（一般来说-450以下是空气的CT值）
    num = -450 # 根据CT图像进行微调
    volarray[volarray>=num]=1
    volarray[volarray<=num]=0
    #生成阈值图像
    threshold = sitk.GetImageFromArray(volarray)
    threshold.SetSpacing(spacing)

    #利用种子生成算法，填充空气
    ConnectedThresholdImageFilter = sitk.ConnectedThresholdImageFilter()
    ConnectedThresholdImageFilter.SetLower(0)
    ConnectedThresholdImageFilter.SetUpper(0)
    ConnectedThresholdImageFilter.SetSeedList([(0,0,0),(size[0]-1,size[1]-1,0)])
    
    #得到body的mask，此时body部分是0，所以反转一下
    bodymask = ConnectedThresholdImageFilter.Execute(threshold)
    bodymask = sitk.ShiftScale(bodymask,-1,-1)
    
    #用bodymask减去threshold，得到初步的lung的mask
    # sitk.Cast(bodymask, sitk.sitkInt16)
    # sitk.Cast(threshold, sitk.sitkInt16)
    temp_array = sitk.GetArrayFromImage(bodymask)-sitk.GetArrayFromImage(threshold)
    # temp_array.dtype = 'int16' #np.int16 # # 512 -> 1024
    
    temp = sitk.GetImageFromArray(temp_array)
    temp = sitk.Cast(temp, sitk.sitkInt16)
    temp.SetSpacing(spacing)
    
    #利用形态学来去掉一定的肺部的小区域
    bm = sitk.BinaryMorphologicalClosingImageFilter()
    bm.SetKernelType(sitk.sitkBall)
    bm.SetKernelRadius(4) # 微调参数可以消除未分割到的肺部小区域
    bm.SetForegroundValue(1)
    lungmask = bm.Execute(temp)  
    
    #利用measure来计算连通域
    lungmaskarray = sitk.GetArrayFromImage(lungmask)
    label = measure.label(lungmaskarray, connectivity=2)
    label1 = copy.deepcopy(label)
    
    # # test : 体素小于100的归零
    # label[label<100]=0

    props = measure.regionprops(label)

    #计算每个连通域的体素的个数
    numPix = []
    for ia in range(len(props)):
        numPix += [props[ia].area]
    
    #最大连通域的体素个数，也就是肺部
    #遍历每个连通区域,增加接口
    index = np.argmax(numPix) # 得到最大的list

    # if remove_block == 1:
    #     numPix[index]=0 # 去掉最大的index
    #     index = np.argmax(numPix)

    label[label!=index+1]=0 
    label[label==index+1]=1
    label = label.astype("int16")

    rows, cols, slices = label.nonzero()
    x = rows.mean()
    y = cols.mean()
    z = slices.mean()

    # 自动判断是否分割成床板
    if y > label.shape[1]*0.667:
        print("need remove block")
        numPix[index]=0 # 去掉最大的index
        index = np.argmax(numPix)
        label1[label1!=index+1]=0 
        label1[label1==index+1]=1
        label = label1.astype("int16")       


    l = sitk.GetImageFromArray(label)
    l.SetSpacing(spacing)
    l.SetOrigin(oringin)
    l.SetDirection(direction)

    # sitk.WriteImage(l, save_path) # 保存图像

    end = time.time()
    print('process end','time:'+str(end-start))

    return l


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


def vesseg(image, label):
    # 窗宽调整
    wintrans = window_transform(image, -1350.0, 650.0)
    # nib.Nifti1Image(wintrans, affine).to_filename(save_path+'wintrans.nii.gz')

    # 获取ROI
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    label = cv2.erode(label, kernel)
    roi = wintrans * label
    # nib.Nifti1Image(roi, affine).to_filename(save_path+'roi.nii.gz')

    # 非线性映射
    roi_sigmoid = sigmoid(roi, 20, 95)
    # nib.Nifti1Image(roi_sigmoid, affine).to_filename(save_path+'sigmoid.nii.gz')

    # 第四步：血管增强
    roi_frangi = frangi(roi_sigmoid, sigmas=range(1, 5, 1),
                                alpha=0.5, beta=0.5, gamma=50, 
                                black_ridges=False, mode='constant', cval=0)
    '''
    para image: 输入图像。
    para sigmas: 滤波器尺度，即 np.arange(scale_range[0], scale_range[1], scale_step)。
    para scale_range：使用后的sigma范围。
    para scale_step：sigma的步长。
    para alptha：Frangi校正常数，用于调整过滤器对于板状结构偏差的敏感度。
    para beta：Frangi校正常数，用于调整过滤器对于斑状结构偏差的敏感度。
    para gamma：Frangi校正常数，用于调整过滤器对高方差/纹理/结构区域的敏感度。
    para black_ridges：当为Ture时，过滤去检测黑色脊线；当为False时，检测白色脊线。
    para mode：可选'constant'、'reflect'、'wrap'、'nearest'、'mirror'五种模式，处理图像边界外的值。
    para cval：与mode的'constant'（图像边界之外的值）结合使用。
    '''
    # nib.Nifti1Image(roi_frangi, affine).to_filename(save_path+'frangi.nii.gz')

    # 第五步：自适应阈值分割
    cv2.normalize(roi_frangi, roi_frangi, 0, 1, cv2.NORM_MINMAX)
    thresh = np.percentile(sorted(roi_frangi[roi_frangi > 0]), 95)
    vessel = (roi_frangi - thresh) * (roi_frangi > thresh) / (1 - thresh)
    vessel[vessel > 0] = 1
    vessel[vessel <= 0] = 0
    return vessel

def vessel_seg(input_path, lung_mask_image, output_path):

    start = time.time()
    print("now start seg vessel!")

    lung_image = sitk.ReadImage(input_path)
    lung_arr = sitk.GetArrayFromImage(lung_image) # zyx
    lung_arr_T = lung_arr.transpose((2, 1, 0)) # zyx => xyz

    # lung_mask_image = sitk.ReadImage(lung_mask_path)
    lung_mask_arr = sitk.GetArrayFromImage(lung_mask_image) # zyx
    lung_mask_arr_T = lung_mask_arr.transpose((2, 1, 0)) # zyx => xyz

    vessel_arr_T = vesseg(lung_arr_T, lung_mask_arr_T)

    vessel_arr = vessel_arr_T.transpose((2, 1, 0)) # xyz => zyx


    vessel_mask_img = sitk.GetImageFromArray(vessel_arr)
    vessel_mask_img.SetDirection(lung_image.GetDirection())
    vessel_mask_img.SetOrigin(lung_image.GetOrigin())
    vessel_mask_img.SetSpacing(lung_image.GetSpacing())

    sitk.WriteImage(vessel_mask_img, output_path)

    end = time.time()
    print('seg vessel process end','time:'+str(end-start))

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
    itk.imwrite(multi_scale_filter.GetOutput(), "vessel_step1.nii.gz")
    # 2
    print("step:2")
    # tme_1 = time.time()
    OutputPixelType = itk.UC
    OutputImageType = itk.Image[OutputPixelType, Dimension]

    rescale_filter = itk.RescaleIntensityImageFilter[ImageType, OutputImageType].New()
    rescale_filter.SetInput(multi_scale_filter)
    # tme_2 = time.time()
    # print('execution time: ' + str(round(tme_2-tme_1))+' s')
    itk.imwrite(rescale_filter.GetOutput(), "vessel_step2.nii.gz")
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
    itk.imwrite(thresholdFilter.GetOutput(), "vessel_step3.nii.gz")
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


# 血管分割V3:
# chatgpt对vessel_seg_V2使用simpleitk进行替换，失败！
def vessel_seg_V3(input_image, output_image, lung_mask_path=None):

    print("血管开始分割")
    sigma_minimum = 2.
    sigma_maximum = 2.
    number_of_sigma_steps = 8
    lowerThreshold = 40
    # output_image = 'vessel'+str(sigma_maximum)+'.nii.gz'

    input_image = sitk.ReadImage(input_image)
    Dimension = input_image.GetDimension()

    objectness_filter = sitk.HessianToObjectnessMeasureImageFilter()
    objectness_filter.SetBrightObject(True)
    objectness_filter.SetScaleObjectnessMeasure(True)
    objectness_filter.SetAlpha(0.5)
    objectness_filter.SetBeta(1.0)
    objectness_filter.SetGamma(5.0)

    multi_scale_filter = sitk.MultiScaleHessianBasedMeasureImageFilter()
    multi_scale_filter.SetInput(input_image)
    multi_scale_filter.SetHessianToMeasureFilter(objectness_filter)
    multi_scale_filter.SetSigmaStepMethodToLogarithmic()
    multi_scale_filter.SetSigmaMinimum(sigma_minimum)
    multi_scale_filter.SetSigmaMaximum(sigma_maximum)
    multi_scale_filter.SetNumberOfSigmaSteps(number_of_sigma_steps)

    OutputPixelType = sitk.sitkUInt8
    rescale_filter = sitk.RescaleIntensityImageFilter()
    rescale_filter.SetOutputMinimum(0)
    rescale_filter.SetOutputMaximum(255)
    rescale_filter.SetOutputPixelType(OutputPixelType)
    rescale_filter.SetInput(multi_scale_filter)

    thresholdFilter = sitk.BinaryThresholdImageFilter()
    thresholdFilter.SetLowerThreshold(lowerThreshold)
    thresholdFilter.SetUpperThreshold(255)
    thresholdFilter.SetOutsideValue(0)
    thresholdFilter.SetInsideValue(255)
    thresholdFilter.SetInput(rescale_filter.GetOutput())

    sitk.WriteImage(thresholdFilter.GetOutput(), output_image)

# chatgpt生成：分割失败，有bug
def vessel_seg_V4(input_image, output_image):
    # 读取输入的CT图像
    input_img = sitk.ReadImage(input_image)

    # 预处理：去除噪声、平滑图像
    input_img_smoothed = sitk.CurvatureFlow(input_img, timeStep=0.125, numberOfIterations=5)

    # 阈值分割肺部组织
    threshold_filter = sitk.BinaryThresholdImageFilter()
    threshold_filter.SetLowerThreshold(-900)
    threshold_filter.SetUpperThreshold(-500)
    threshold_filter.SetInsideValue(0)
    threshold_filter.SetOutsideValue(1)

    lung_mask = threshold_filter.Execute(input_img_smoothed)

    # 进一步处理，去除肺部周围的组织，得到纯净的肺部组织掩膜
    connected_component_filter = sitk.ConnectedComponentImageFilter()
    lung_mask_cc = connected_component_filter.Execute(lung_mask)

    lung_mask_stats = sitk.LabelShapeStatisticsImageFilter()
    lung_mask_stats.Execute(lung_mask_cc)

    all_labels = lung_mask_stats.GetLabels()

    if len(all_labels) > 1:
        label_sizes = [lung_mask_stats.GetNumberOfPixels(l) for l in all_labels]

        largest_label_idx = label_sizes.index(max(label_sizes))
        label = all_labels[largest_label_idx]
    else:
        label = all_labels[0]

    label_statistics_filter = sitk.LabelShapeStatisticsImageFilter()
    label_statistics_filter.Execute(lung_mask_cc)
    bounding_box = label_statistics_filter.GetBoundingBox(1)
    
    lung_mask_cc = sitk.RegionOfInterest(lung_mask_cc, bounding_box)
    print(lung_mask_cc.GetLargestPossibleRegion().GetSize())
    print(lung_mask_cc.GetBufferedRegion().GetSize())
    print(lung_mask_cc.GetRequestedRegion().GetSize())
    sitk.WriteImage(lung_mask_cc, 'lung_mask_cc.nii.gz')
    print("肺部掩膜分割完毕！")

    # 将肺部组织掩膜和血管掩膜的空间域匹配
    # lung_mask = resample_image(lung_mask, vessel_mask)

    # # 计算血管掩膜
    # vessel_filter = sitk.BinaryDilateImageFilter()
    # vessel_filter.SetKernelType(sitk.sitkBall)
    # vessel_filter.SetKernelRadius(1)
    # vessel_mask = vessel_filter.Execute(lung_mask_cc)
    # vessel_mask = sitk.Cast(vessel_mask, sitk.sitkUInt8)

    # # lung_mask = resample_image(lung_mask, vessel_mask)
    # # vessel_mask = lung_mask & ~vessel_mask

    # # 保存输出结果
    # sitk.WriteImage(vessel_mask, output_image)
    # print("血管掩膜已保存成功！")

# 定义空间域匹配函数
def resample_image(image, reference_image):
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_image)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetOutputPixelType(sitk.sitkUInt8)
    resampled_image = resampler.Execute(image)
    return resampled_image

############################### 气管与肺分割 #########################
def airways_lung_seg(input_path, seed, lung_mask_save_path, airways_mask_save_path, lung_airways_save_path=None, remove_block=None):
    '''
    params input_path: 输入肺部CT的nii.gz格式
    params seeds: 种子点，一个三个，seeds[0]左肺 seeds[1]右肺 seeds[2]气管
    params lung_mask_save_path: 肺掩膜保存路径
    params airways_mask_save_path： 气管掩膜保存路径
    '''
    airways_image = airways_seg(image_path = input_path,
                output_path = airways_mask_save_path,
                seed = seed)

    # lung_image = lung_cu_seg(img_path=input_path, save_path=None, seed_pts = seeds[0:2])
    lung_image = lung_cu_seg_v2(img_path=input_path, remove_block=remove_block)

    lung_arr = sitk.GetArrayFromImage(lung_image)


    airways_arr = sitk.GetArrayFromImage(airways_image)


    lung_arr[airways_arr==1] = 0

    lung_mask_image = sitk.GetImageFromArray(lung_arr)
    lung_mask_image.SetSpacing(lung_image.GetSpacing())
    lung_mask_image.SetDirection(lung_image.GetDirection())
    lung_mask_image.SetOrigin(lung_image.GetOrigin())

    # 开运算去毛刺
    lung_mask_image = sitk.BinaryMorphologicalOpening(lung_mask_image != 0, [5,5,5])

    # 保留最大两个联通区域，防止多余的杂质。
    # lung_mask_image = Get2LargestConnectedCompont(lung_mask_image)

    sitk.WriteImage(lung_mask_image, lung_mask_save_path) # 保存细分割的肺

    if lung_airways_save_path != None:
        sitk.WriteImage(lung_image, lung_airways_save_path) # 保存粗分割的肺与气管

    # return lung_image # 返回粗分割的肺，用于血管分割



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

    sitk.WriteImage(image1,"skin1.nii.gz")

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

    sitk.WriteImage(outmasksitk,"skin2.nii.gz")

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

#################################骨头分割###############################
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
    print("bone开始分割")
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
    print("bone seg over")






if __name__ == '__main__':
    num = 15
    # input_path = "D:\\workplace\\hospital\\dataset#pig\\pig4.nii.gz"# './dataset/LIDC-IDRI-00'+str(num)+'.nii.gz'
    input_path = "lung_001_0000.nii.gz"
    # input_path = 'pig_window_intensity.nii.gz'
    # makefile = (input_path.split('\\')[-1]).split('.')[0]
    makefile = input_path.split('.')[0]

    pig = 0
    if pig:
        input_image = input_path
        sitk.WriteImage(
            window_intensity(
            sitk.ReadImage(input_image), -1024, 2048), 
            input_image)

    import os

    makefile = os.path.join('output', makefile)
    os.makedirs(makefile, exist_ok=True)
    seed = [247, 251, 228]  #16[256, 258, 283]  # 24[259, 190, 287]寄了  17[279, 237, 300]  12[259,228,282]
    bone_lowerThreshold = 129 # 129为000最佳 对于016 017 

    
    lung_airways_save_path = os.path.join(makefile, 'lung_airways_mask.nii.gz') #'./all_output/lung_airways_mask_'+str(num)+'.nii.gz' # 肺气管粗分割路径 

    lung_mask_save_path = os.path.join(makefile, 'lung_mask.nii.gz') #'./all_output/lung_mask_{:02d}.nii.gz'.format(num) # 肺掩膜保存路径
    airways_mask_save_path = os.path.join(makefile, "airways_mask.nii.gz")  #'./all_output/airways_mask_'+str(num)+'.nii.gz' # 气管掩膜保存路径
    vessel_mask_save_path = os.path.join(makefile, 'vessel_mask.nii.gz') # 血管掩膜保存路径

    

    skin_mask_save_path = os.path.join(makefile, 'skin_mask.nii.gz') # 皮肤掩膜保存路径

    skeleton_save_path = os.path.join(makefile, 'skeleton.nii.gz') # 骨骼掩膜保存路径

    # ## 气管 todo: 动态阈值
    # # airways_image = airways_seg(image_path = input_path,
    # #             output_path = airways_mask_save_path,
    # #             seed = seed, threshold=113, open_grow_mode = True)
  
    # # 气管与肺
    airways_lung_seg(input_path, seed, lung_mask_save_path, 
                                        airways_mask_save_path, lung_airways_save_path, remove_block=0)  # remove_block 判断是都有床板情况

    # # # 血管
    vessel_seg_V2(input_path, vessel_mask_save_path, lung_airways_save_path, pig=0) #可以使用肺气管粗分割掩膜


    # #皮肤
    skin_mask(input_path, lung_airways_save_path, skin_mask_save_path)


    # # # 骨骼
    skeletonSegment(input_path, skeleton_save_path, lowerThreshold=bone_lowerThreshold)



    # input_path = "D:\\test_dataset\\opensource\\imagesTr_redirection\\lung_043.nii.gz"
    # skeleton_save_path = "D:\\test_dataset\\opensource\\new_label\\043\\bone.nii.gz"
    # skeletonSegment(input_path, skeleton_save_path, lowerThreshold=150)