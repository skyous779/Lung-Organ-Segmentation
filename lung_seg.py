import SimpleITK as sitk 
import numpy as np
import time
from skimage import measure
import copy


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
