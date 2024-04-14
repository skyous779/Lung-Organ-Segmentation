import SimpleITK as sitk
import numpy as np
import time
from utils import *

from skimage import measure


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

