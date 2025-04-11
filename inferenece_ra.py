import os
import time
from mmdet.apis import DetInferencer  # 确保你用的是正确的推理器类

data_dir = 'original/'  # 数据集路径
output_dir = 'laf_vis_epoch446/'  # 保存可视化结果的目录
model_file = '../../../configs/mm_grounding_dino/grounding_dino_swin-t_finetune_8xb4_20e_cat.py'
checkpoint_file = '../../../work_dir/roadanomalyv2_work_dir/best_coco_bbox_mAP_epoch_446.pth'
texts = 'ood-object'

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 初始化推理器(只加载一次模型)
inferencer = DetInferencer(
    model=model_file,
    weights=checkpoint_file,
    device='cuda:0',
    palette='coco'
)

# 遍历数据集目录中的每张图片
total_time = 0
image_count = 0

for img_filename in os.listdir(data_dir):
    if not img_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
        continue

    img_path = os.path.join(data_dir, img_filename)
    
    start_time = time.time()
    
    # 直接使用已加载的模型进行推理
    inferencer(
        inputs=img_path,
        texts=texts,
        out_dir=output_dir,
        pred_score_thr=0.7,
        no_save_pred=False,
    )
    
    end_time = time.time()
    
    image_count += 1
    time_taken = end_time - start_time
    total_time += time_taken
    
    print(f'Processed {img_filename} - Time: {time_taken:.4f} sec')

# 计算 FPS
if image_count > 0:
    fps = image_count / total_time
    print(f"\nTotal images processed: {image_count}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average FPS: {fps:.2f}")
else:
    print("No valid images found for processing.")
