import os
from pathlib import Path


def save_evaluation_results(output_dir, phase_idx, test_stats, coco_evaluator, suffix=''):
    if not output_dir:
        return
        
    output_dir = Path(output_dir)
    result_dir = output_dir / 'results'
    result_dir.mkdir(parents=True, exist_ok=True)
    
    stats_file = f'test_stats_phase_{phase_idx}{suffix}.txt'
    
    try:
        with open(result_dir / stats_file, 'w') as f:
            if coco_evaluator is not None:
                for iou_type in coco_evaluator.coco_eval:
                    stats = coco_evaluator.coco_eval[iou_type].stats
                    f.write(f"IoU metric: {iou_type}\n")
                    f.write(f" Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {stats[0]:.3f}\n")
                    f.write(f" Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = {stats[1]:.3f}\n")
                    f.write(f" Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = {stats[2]:.3f}\n")
                    f.write(f" Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = {stats[3]:.3f}\n")
                    f.write(f" Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = {stats[4]:.3f}\n")
                    f.write(f" Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = {stats[5]:.3f}\n")
                    f.write(f" Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = {stats[6]:.3f}\n")
                    f.write(f" Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = {stats[7]:.3f}\n")
                    f.write(f" Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {stats[8]:.3f}\n")
                    f.write(f" Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = {stats[9]:.3f}\n")
                    f.write(f" Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = {stats[10]:.3f}\n")
                    f.write(f" Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = {stats[11]:.3f}\n")
            
            print(f"测试结果已保存到 {result_dir}/{stats_file}")
            
    except Exception as e:
        print(f"保存评估结果时出错: {e}")
