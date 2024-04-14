import os

from all_seg_demo import airways_lung_seg
from vessel_seg import vessel_seg_V2
from skin_seg import skin_mask
from skeleton_seg import skeletonSegment
    
input_path = "lung_001_0000.nii.gz"

makefile = input_path.split('.')[0]

makefile = os.path.join('output', makefile)
os.makedirs(makefile, exist_ok=True)
seed = [247, 251, 228]  #16[256, 258, 283]  # 24[259, 190, 287]寄了  17[279, 237, 300]  12[259,228,282]
bone_lowerThreshold = 129 # 129为000最佳 对于016 017 

lung_airways_save_path = os.path.join(makefile, 'lung_airways_mask.nii.gz')

lung_airways_save_path = os.path.join(makefile, 'lung_airways_mask.nii.gz') #'./all_output/lung_airways_mask_'+str(num)+'.nii.gz' # 肺气管粗分割路径 

lung_mask_save_path = os.path.join(makefile, 'lung_mask.nii.gz') #'./all_output/lung_mask_{:02d}.nii.gz'.format(num) # 肺掩膜保存路径
airways_mask_save_path = os.path.join(makefile, "airways_mask.nii.gz")  #'./all_output/airways_mask_'+str(num)+'.nii.gz' # 气管掩膜保存路径
vessel_mask_save_path = os.path.join(makefile, 'vessel_mask.nii.gz') # 血管掩膜保存路径

skin_mask_save_path = os.path.join(makefile, 'skin_mask.nii.gz') # 皮肤掩膜保存路径
skeleton_save_path = os.path.join(makefile, 'skeleton.nii.gz') # 骨骼掩膜保存路径


airways_lung_seg(input_path, seed, lung_mask_save_path, 
                 airways_mask_save_path, lung_airways_save_path, remove_block=0)  # remove_block 判断是都有床板情况

   


# # # 血管
vessel_seg_V2(input_path, vessel_mask_save_path, lung_airways_save_path, pig=0) #可以使用肺气管粗分割掩膜

# #皮肤
skin_mask(input_path, lung_airways_save_path, skin_mask_save_path)

# # # 骨骼
skeletonSegment(input_path, skeleton_save_path, lowerThreshold=bone_lowerThreshold)

