# Official YOLOv7

Implementation of paper - [YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors](https://arxiv.org/abs/2207.02696)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/yolov7-trainable-bag-of-freebies-sets-new/real-time-object-detection-on-coco)](https://paperswithcode.com/sota/real-time-object-detection-on-coco?p=yolov7-trainable-bag-of-freebies-sets-new)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/akhaliq/yolov7)
<a href="https://colab.research.google.com/gist/AlexeyAB/b769f5795e65fdab80086f6cb7940dae/yolov7detection.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
[![arxiv.org](http://img.shields.io/badge/cs.CV-arXiv%3A2207.02696-B31B1B.svg)](https://arxiv.org/abs/2207.02696)

<div align="center">
    <a href="./">
        <img src="./figure/performance.png" width="79%"/>
    </a>
</div>

## Web Demo

- Integrated into [Huggingface Spaces 🤗](https://huggingface.co/spaces/akhaliq/yolov7) using Gradio. Try out the Web Demo [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/akhaliq/yolov7)

## Performance 

MS COCO

| Model | Test Size | AP<sup>test</sup> | AP<sub>50</sub><sup>test</sup> | AP<sub>75</sub><sup>test</sup> | batch 1 fps | batch 32 average time |
| :-- | :-: | :-: | :-: | :-: | :-: | :-: |
| [**YOLOv7**](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt) | 640 | **51.4%** | **69.7%** | **55.9%** | 161 *fps* | 2.8 *ms* |
| [**YOLOv7-X**](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7x.pt) | 640 | **53.1%** | **71.2%** | **57.8%** | 114 *fps* | 4.3 *ms* |
|  |  |  |  |  |  |  |
| [**YOLOv7-W6**](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6.pt) | 1280 | **54.9%** | **72.6%** | **60.1%** | 84 *fps* | 7.6 *ms* |
| [**YOLOv7-E6**](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6.pt) | 1280 | **56.0%** | **73.5%** | **61.2%** | 56 *fps* | 12.3 *ms* |
| [**YOLOv7-D6**](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-d6.pt) | 1280 | **56.6%** | **74.0%** | **61.8%** | 44 *fps* | 15.0 *ms* |
| [**YOLOv7-E6E**](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e.pt) | 1280 | **56.8%** | **74.4%** | **62.1%** | 36 *fps* | 18.7 *ms* |

## Installation

Docker environment (recommended)
<details><summary> <b>Expand</b> </summary>

``` shell
# create the docker container, you can change the share memory size if you have more.
nvidia-docker run --name yolov7 -it -v your_coco_path/:/coco/ -v your_code_path/:/yolov7 --shm-size=64g nvcr.io/nvidia/pytorch:21.08-py3

# apt install required packages
apt update
apt install -y zip htop screen libgl1-mesa-glx

# pip install required packages
pip install seaborn thop

# go to code folder
cd /yolov7
```

</details>

## Testing

[`yolov7.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt) [`yolov7x.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7x.pt) [`yolov7-w6.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6.pt) [`yolov7-e6.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6.pt) [`yolov7-d6.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-d6.pt) [`yolov7-e6e.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e.pt)

``` shell
python test.py --data data/coco.yaml --img 640 --batch 32 --conf 0.001 --iou 0.65 --device 0 --weights yolov7.pt --name yolov7_640_val
```

You will get the results:

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.51206
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.69730
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.55521
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.35247
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.55937
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.66693
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.38453
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.63765
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.68772
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.53766
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.73549
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.83868
```

To measure accuracy, download [COCO-annotations for Pycocotools](http://images.cocodataset.org/annotations/annotations_trainval2017.zip) to the `./coco/annotations/instances_val2017.json`

## Training

Data preparation

``` shell
bash scripts/get_coco.sh
```

* Download MS COCO dataset images ([train](http://images.cocodataset.org/zips/train2017.zip), [val](http://images.cocodataset.org/zips/val2017.zip), [test](http://images.cocodataset.org/zips/test2017.zip)) and [labels](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/coco2017labels-segments.zip). If you have previously used a different version of YOLO, we strongly recommend that you delete `train2017.cache` and `val2017.cache` files, and redownload [labels](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/coco2017labels-segments.zip) 

Single GPU training

``` shell
# train p5 models
python train.py --workers 8 --device 0 --batch-size 32 --data data/coco.yaml --img 640 640 --cfg cfg/training/yolov7.yaml --weights '' --name yolov7 --hyp data/hyp.scratch.p5.yaml

# train p6 models
python train_aux.py --workers 8 --device 0 --batch-size 16 --data data/coco.yaml --img 1280 1280 --cfg cfg/training/yolov7-w6.yaml --weights '' --name yolov7-w6 --hyp data/hyp.scratch.p6.yaml
```

Multiple GPU training

``` shell
# train p5 models
python -m torch.distributed.launch --nproc_per_node 4 --master_port 9527 train.py --workers 8 --device 0,1,2,3 --sync-bn --batch-size 128 --data data/coco.yaml --img 640 640 --cfg cfg/training/yolov7.yaml --weights '' --name yolov7 --hyp data/hyp.scratch.p5.yaml

# train p6 models
python -m torch.distributed.launch --nproc_per_node 8 --master_port 9527 train_aux.py --workers 8 --device 0,1,2,3,4,5,6,7 --sync-bn --batch-size 128 --data data/coco.yaml --img 1280 1280 --cfg cfg/training/yolov7-w6.yaml --weights '' --name yolov7-w6 --hyp data/hyp.scratch.p6.yaml
```

## Transfer learning

[`yolov7_training.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7_training.pt) [`yolov7x_training.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7x_training.pt) [`yolov7-w6_training.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6_training.pt) [`yolov7-e6_training.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6_training.pt) [`yolov7-d6_training.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-d6_training.pt) [`yolov7-e6e_training.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e_training.pt)

Single GPU finetuning for custom dataset

``` shell
# finetune p5 models
python train.py --workers 8 --device 0 --batch-size 32 --data data/custom.yaml --img 640 640 --cfg cfg/training/yolov7-custom.yaml --weights 'yolov7_training.pt' --name yolov7-custom --hyp data/hyp.scratch.custom.yaml

# finetune p6 models
python train_aux.py --workers 8 --device 0 --batch-size 16 --data data/custom.yaml --img 1280 1280 --cfg cfg/training/yolov7-w6-custom.yaml --weights 'yolov7-w6_training.pt' --name yolov7-w6-custom --hyp data/hyp.scratch.custom.yaml
```

## Re-parameterization

See [reparameterization.ipynb](tools/reparameterization.ipynb)

## Inference

On video:
``` shell
python detect.py --weights yolov7.pt --conf 0.25 --img-size 640 --source yourvideo.mp4
```

On image:
``` shell
python detect.py --weights yolov7.pt --conf 0.25 --img-size 640 --source inference/images/horses.jpg
```

<div align="center">
    <a href="./">
        <img src="./figure/horses_prediction.jpg" width="59%"/>
    </a>
</div>


## Export

**Pytorch to CoreML (and inference on MacOS/iOS)** <a href="https://colab.research.google.com/github/WongKinYiu/yolov7/blob/main/tools/YOLOv7CoreML.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>

**Pytorch to ONNX with NMS (and inference)** <a href="https://colab.research.google.com/github/WongKinYiu/yolov7/blob/main/tools/YOLOv7onnx.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
```shell
python export.py --weights yolov7-tiny.pt --grid --end2end --simplify \
        --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size 640 640 --max-wh 640
```

**Pytorch to TensorRT with NMS (and inference)** <a href="https://colab.research.google.com/github/WongKinYiu/yolov7/blob/main/tools/YOLOv7trt.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>

```shell
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt
python export.py --weights ./yolov7-tiny.pt --grid --end2end --simplify --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size 640 640
git clone https://github.com/Linaom1214/tensorrt-python.git
python ./tensorrt-python/export.py -o yolov7-tiny.onnx -e yolov7-tiny-nms.trt -p fp16
```

**Pytorch to TensorRT another way** <a href="https://colab.research.google.com/gist/AlexeyAB/fcb47ae544cf284eb24d8ad8e880d45c/yolov7trtlinaom.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a> <details><summary> <b>Expand</b> </summary>


```shell
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt
python export.py --weights yolov7-tiny.pt --grid --include-nms
git clone https://github.com/Linaom1214/tensorrt-python.git
python ./tensorrt-python/export.py -o yolov7-tiny.onnx -e yolov7-tiny-nms.trt -p fp16

# Or use trtexec to convert ONNX to TensorRT engine
/usr/src/tensorrt/bin/trtexec --onnx=yolov7-tiny.onnx --saveEngine=yolov7-tiny-nms.trt --fp16
```

</details>

Tested with: Python 3.7.13, Pytorch 1.12.0+cu113

## Pose estimation

[`code`](https://github.com/WongKinYiu/yolov7/tree/pose) [`yolov7-w6-pose.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-pose.pt)

See [keypoint.ipynb](https://github.com/WongKinYiu/yolov7/blob/main/tools/keypoint.ipynb).

<div align="center">
    <a href="./">
        <img src="./figure/pose.png" width="39%"/>
    </a>
</div>


## Instance segmentation (with NTU)

[`code`](https://github.com/WongKinYiu/yolov7/tree/mask) [`yolov7-mask.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-mask.pt)

See [instance.ipynb](https://github.com/WongKinYiu/yolov7/blob/main/tools/instance.ipynb).

<div align="center">
    <a href="./">
        <img src="./figure/mask.png" width="59%"/>
    </a>
</div>

## Instance segmentation

[`code`](https://github.com/WongKinYiu/yolov7/tree/u7/seg) [`yolov7-seg.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-seg.pt)

YOLOv7 for instance segmentation (YOLOR + YOLOv5 + YOLACT)

| Model | Test Size | AP<sup>box</sup> | AP<sub>50</sub><sup>box</sup> | AP<sub>75</sub><sup>box</sup> | AP<sup>mask</sup> | AP<sub>50</sub><sup>mask</sup> | AP<sub>75</sub><sup>mask</sup> |
| :-- | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| **YOLOv7-seg** | 640 | **51.4%** | **69.4%** | **55.8%** | **41.5%** | **65.5%** | **43.7%** |

## Anchor free detection head

[`code`](https://github.com/WongKinYiu/yolov7/tree/u6) [`yolov7-u6.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-u6.pt)

YOLOv7 with decoupled TAL head (YOLOR + YOLOv5 + YOLOv6)

| Model | Test Size | AP<sup>val</sup> | AP<sub>50</sub><sup>val</sup> | AP<sub>75</sub><sup>val</sup> |
| :-- | :-: | :-: | :-: | :-: |
| **YOLOv7-u6** | 640 | **52.6%** | **69.7%** | **57.3%** |


## Citation

```
@inproceedings{wang2023yolov7,
  title={{YOLOv7}: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors},
  author={Wang, Chien-Yao and Bochkovskiy, Alexey and Liao, Hong-Yuan Mark},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2023}
}
```

```
@article{wang2023designing,
  title={Designing Network Design Strategies Through Gradient Path Analysis},
  author={Wang, Chien-Yao and Liao, Hong-Yuan Mark and Yeh, I-Hau},
  journal={Journal of Information Science and Engineering},
  year={2023}
}
```


## Teaser

YOLOv7-semantic & YOLOv7-panoptic & YOLOv7-caption

<div align="center">
    <a href="./">
        <img src="./figure/tennis.jpg" width="24%"/>
    </a>
    <a href="./">
        <img src="./figure/tennis_semantic.jpg" width="24%"/>
    </a>
    <a href="./">
        <img src="./figure/tennis_panoptic.png" width="24%"/>
    </a>
    <a href="./">
        <img src="./figure/tennis_caption.png" width="24%"/>
    </a>
</div>

YOLOv7-semantic & YOLOv7-detection & YOLOv7-depth (with NTUT)

<div align="center">
    <a href="./">
        <img src="./figure/yolov7_city.jpg" width="80%"/>
    </a>
</div>

YOLOv7-3d-detection & YOLOv7-lidar & YOLOv7-road (with NTUT)

<div align="center">
    <a href="./">
        <img src="./figure/yolov7_3d.jpg" width="30%"/>
    </a>
    <a href="./">
        <img src="./figure/yolov7_lidar.jpg" width="30%"/>
    </a>
    <a href="./">
        <img src="./figure/yolov7_road.jpg" width="30%"/>
    </a>
</div>


## Acknowledgements

<details><summary> <b>Expand</b> </summary>

* [https://github.com/AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)
* [https://github.com/WongKinYiu/yolor](https://github.com/WongKinYiu/yolor)
* [https://github.com/WongKinYiu/PyTorch_YOLOv4](https://github.com/WongKinYiu/PyTorch_YOLOv4)
* [https://github.com/WongKinYiu/ScaledYOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4)
* [https://github.com/Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
* [https://github.com/ultralytics/yolov3](https://github.com/ultralytics/yolov3)
* [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
* [https://github.com/DingXiaoH/RepVGG](https://github.com/DingXiaoH/RepVGG)
* [https://github.com/JUGGHM/OREPA_CVPR2022](https://github.com/JUGGHM/OREPA_CVPR2022)
* [https://github.com/TexasInstruments/edgeai-yolov5/tree/yolo-pose](https://github.com/TexasInstruments/edgeai-yolov5/tree/yolo-pose)

</details>

## Reference
* [Install TensorRT](https://blog.csdn.net/lkh798842577/article/details/120218588)

## Log
<details><summary> <b>Expand</b> </summary>

* 20240108
```
(yolov7-TensorRT) ytl0623@lab508:~/data/yolov7-TensorRT$ python ./tensorrt-python/export.py -o yolov7-tiny.onnx -e yolov7-tiny-nms.trt -p fp16
Namespace(calib_batch_size=8, calib_cache='./calibration.cache', calib_input=None, calib_num_images=5000, conf_thres=0.4, end2end=False, engine='yolov7-tiny-nms.trt', iou_thres=0.5, max_det=100, onnx='yolov7-tiny.onnx', precision='fp16', v8=False, verbose=False, workspace=1)
[01/08/2024-17:06:37] [TRT] [W] Unable to determine GPU memory usage
[01/08/2024-17:06:37] [TRT] [W] Unable to determine GPU memory usage
[01/08/2024-17:06:37] [TRT] [I] [MemUsageChange] Init CUDA: CPU +0, GPU +0, now: CPU 23, GPU 0 (MiB)
[01/08/2024-17:06:37] [TRT] [W] CUDA initialization failure with error: 35. Please check your CUDA installation:  http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html
Traceback (most recent call last):
  File "./tensorrt-python/export.py", line 308, in <module>
    main(args)
  File "./tensorrt-python/export.py", line 266, in main
    builder = EngineBuilder(args.verbose, args.workspace)
  File "./tensorrt-python/export.py", line 107, in __init__
    self.builder = trt.Builder(self.trt_logger)
TypeError: pybind11::init(): factory function returned nullptr
(yolov7-TensorRT) ytl0623@lab508:~/data/yolov7-TensorRT$ 
```
* 20240109 Tested with: Windows 11(Anaconda), Python 3.10, Pytorch 2.1.2(Cuda 12.1), TensorRT 8.6.1
```
(base) C:\Users\User>conda activate tensorrt

(tensorrt) C:\Users\User>cd Downloads

(tensorrt) C:\Users\User\Downloads>cd yolov7-TensorRT

(tensorrt) C:\Users\User\Downloads\yolov7-TensorRT>python detect.py --weights yolov7.pt --conf 0.25 --img-size 640 --source inference/images/horses.jpg
Namespace(weights=['yolov7.pt'], source='inference/images/horses.jpg', img_size=640, conf_thres=0.25, iou_thres=0.45, device='', view_img=False, save_txt=False, save_conf=False, nosave=False, classes=None, agnostic_nms=False, augment=False, update=False, project='runs/detect', name='exp', exist_ok=False, no_trace=False)
YOLOR  a207844 torch 2.1.2+cu121 CUDA:0 (NVIDIA GeForce RTX 2060, 6143.6875MB)

Fusing layers...
RepConv.fuse_repvgg_block
RepConv.fuse_repvgg_block
RepConv.fuse_repvgg_block
C:\Users\User\anaconda3\envs\tensorrt\lib\site-packages\torch\functional.py:504: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at ..\aten\src\ATen\native\TensorShape.cpp:3527.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
Model Summary: 306 layers, 36905341 parameters, 6652669 gradients, 104.5 GFLOPS
 Convert model to Traced-model...
 traced_script_module saved!
 model is traced!

5 horses, Done. (13.0ms) Inference, (385.2ms) NMS
 The image with the result is saved in: runs\detect\exp2\horses.jpg
Done. (0.763s)

(tensorrt) C:\Users\User\Downloads\yolov7-TensorRT>python export.py --weights ./yolov7.pt --grid --end2end --simplify --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size 640 640
Namespace(weights='./yolov7.pt', img_size=[640, 640], batch_size=1, dynamic=False, dynamic_batch=False, grid=True, end2end=True, max_wh=None, topk_all=100, iou_thres=0.65, conf_thres=0.35, device='cpu', simplify=True, include_nms=False, fp16=False, int8=False)
YOLOR  a207844 torch 2.1.2+cu121 CPU

Fusing layers...
RepConv.fuse_repvgg_block
RepConv.fuse_repvgg_block
RepConv.fuse_repvgg_block
C:\Users\User\anaconda3\envs\tensorrt\lib\site-packages\torch\functional.py:504: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at ..\aten\src\ATen\native\TensorShape.cpp:3527.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
Model Summary: 306 layers, 36905341 parameters, 36905341 gradients, 104.5 GFLOPS

Starting TorchScript export with torch 2.1.2+cu121...
C:\Users\User\Downloads\yolov7-TensorRT\models\yolo.py:52: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  if self.grid[i].shape[2:4] != x[i].shape[2:4]:
TorchScript export success, saved as ./yolov7.torchscript.pt
CoreML export failure: No module named 'coremltools'

Starting TorchScript-Lite export with torch 2.1.2+cu121...
TorchScript-Lite export success, saved as ./yolov7.torchscript.ptl

Starting ONNX export with onnx 1.15.0...

Starting export end2end onnx model for TensorRT...
C:\Users\User\anaconda3\envs\tensorrt\lib\site-packages\torch\nn\modules\module.py:844: UserWarning: The .grad attribute of a Tensor that is not a leaf Tensor is being accessed. Its .grad attribute won't be populated during autograd.backward(). If you indeed want the .grad field to be populated for a non-leaf Tensor, use .retain_grad() on the non-leaf Tensor. If you access the non-leaf Tensor by mistake, make sure you access the leaf Tensor instead. See github.com/pytorch/pytorch/pull/30531 for more informations. (Triggered internally at aten\src\ATen/core/TensorBody.h:494.)
  if param.grad is not None:
[W shape_type_inference.cpp:1978] Warning: The shape inference of TRT::EfficientNMS_TRT type is missing, so it may result in wrong shape inference for the exported graph. Please consider adding it in symbolic function. (function UpdateReliable)
[W shape_type_inference.cpp:1978] Warning: The shape inference of TRT::EfficientNMS_TRT type is missing, so it may result in wrong shape inference for the exported graph. Please consider adding it in symbolic function. (function UpdateReliable)
[W shape_type_inference.cpp:1978] Warning: The shape inference of TRT::EfficientNMS_TRT type is missing, so it may result in wrong shape inference for the exported graph. Please consider adding it in symbolic function. (function UpdateReliable)
[W shape_type_inference.cpp:1978] Warning: The shape inference of TRT::EfficientNMS_TRT type is missing, so it may result in wrong shape inference for the exported graph. Please consider adding it in symbolic function. (function UpdateReliable)
Simplifier failure: No module named 'onnxsim'
ONNX export success, saved as ./yolov7.onnx

Export complete (19.68s). Visualize with https://github.com/lutzroeder/netron.

(tensorrt) C:\Users\User\Downloads\yolov7-TensorRT>pip install coremltools
Collecting coremltools
  Downloading coremltools-7.1.tar.gz (1.2 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.2/1.2 MB 1.4 MB/s eta 0:00:00
  Preparing metadata (setup.py) ... done
Requirement already satisfied: numpy>=1.14.5 in c:\users\user\anaconda3\envs\tensorrt\lib\site-packages (from coremltools) (1.23.5)
Collecting protobuf<=4.0.0,>=3.1.0 (from coremltools)
  Downloading protobuf-3.20.3-cp310-cp310-win_amd64.whl (904 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 904.0/904.0 kB 2.2 MB/s eta 0:00:00
Requirement already satisfied: sympy in c:\users\user\anaconda3\envs\tensorrt\lib\site-packages (from coremltools) (1.12)
Requirement already satisfied: tqdm in c:\users\user\anaconda3\envs\tensorrt\lib\site-packages (from coremltools) (4.66.1)
Requirement already satisfied: packaging in c:\users\user\anaconda3\envs\tensorrt\lib\site-packages (from coremltools) (23.2)
Collecting attrs>=21.3.0 (from coremltools)
  Downloading attrs-23.2.0-py3-none-any.whl.metadata (9.5 kB)
Collecting cattrs (from coremltools)
  Downloading cattrs-23.2.3-py3-none-any.whl.metadata (10 kB)
Collecting pyaml (from coremltools)
  Downloading pyaml-23.12.0-py3-none-any.whl.metadata (11 kB)
Requirement already satisfied: exceptiongroup>=1.1.1 in c:\users\user\anaconda3\envs\tensorrt\lib\site-packages (from cattrs->coremltools) (1.2.0)
Requirement already satisfied: typing-extensions!=4.6.3,>=4.1.0 in c:\users\user\anaconda3\envs\tensorrt\lib\site-packages (from cattrs->coremltools) (4.9.0)
Requirement already satisfied: PyYAML in c:\users\user\anaconda3\envs\tensorrt\lib\site-packages (from pyaml->coremltools) (6.0.1)
Requirement already satisfied: mpmath>=0.19 in c:\users\user\anaconda3\envs\tensorrt\lib\site-packages (from sympy->coremltools) (1.3.0)
Requirement already satisfied: colorama in c:\users\user\anaconda3\envs\tensorrt\lib\site-packages (from tqdm->coremltools) (0.4.6)
Downloading attrs-23.2.0-py3-none-any.whl (60 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 60.8/60.8 kB 1.6 MB/s eta 0:00:00
Downloading cattrs-23.2.3-py3-none-any.whl (57 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 57.5/57.5 kB 3.0 MB/s eta 0:00:00
Downloading pyaml-23.12.0-py3-none-any.whl (23 kB)
Building wheels for collected packages: coremltools
  Building wheel for coremltools (setup.py) ... done
  Created wheel for coremltools: filename=coremltools-7.1-py3-none-any.whl size=1528749 sha256=b73a46df1d862fb66621fbf9725e98d8eb6489e4759587b1d55c0b1c228a099f
  Stored in directory: c:\users\user\appdata\local\pip\cache\wheels\23\e7\03\7e9a6ca0e734d469cdc121ab36c4b9e17b403d119daf4fd7d5
Successfully built coremltools
Installing collected packages: pyaml, protobuf, attrs, cattrs, coremltools
  Attempting uninstall: protobuf
    Found existing installation: protobuf 4.21.2
    Uninstalling protobuf-4.21.2:
      Successfully uninstalled protobuf-4.21.2
Successfully installed attrs-23.2.0 cattrs-23.2.3 coremltools-7.1 protobuf-3.20.3 pyaml-23.12.0

(tensorrt) C:\Users\User\Downloads\yolov7-TensorRT>python export.py --weights ./yolov7.pt --grid --end2end --simplify --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size 640 640
Namespace(weights='./yolov7.pt', img_size=[640, 640], batch_size=1, dynamic=False, dynamic_batch=False, grid=True, end2end=True, max_wh=None, topk_all=100, iou_thres=0.65, conf_thres=0.35, device='cpu', simplify=True, include_nms=False, fp16=False, int8=False)
YOLOR  a207844 torch 2.1.2+cu121 CPU

Fusing layers...
RepConv.fuse_repvgg_block
RepConv.fuse_repvgg_block
RepConv.fuse_repvgg_block
C:\Users\User\anaconda3\envs\tensorrt\lib\site-packages\torch\functional.py:504: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at ..\aten\src\ATen\native\TensorShape.cpp:3527.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
Model Summary: 306 layers, 36905341 parameters, 36905341 gradients, 104.5 GFLOPS

Starting TorchScript export with torch 2.1.2+cu121...
C:\Users\User\Downloads\yolov7-TensorRT\models\yolo.py:52: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  if self.grid[i].shape[2:4] != x[i].shape[2:4]:
TorchScript export success, saved as ./yolov7.torchscript.pt
Torch version 2.1.2+cu121 has not been tested with coremltools. You may run into unexpected errors. Torch 2.1.0 is the most recent version that has been tested.
Fail to import BlobReader from libmilstoragepython. No module named 'coremltools.libmilstoragepython'
Fail to import BlobWriter from libmilstoragepython. No module named 'coremltools.libmilstoragepython'

Starting CoreML export with coremltools 7.1...
When both 'convert_to' and 'minimum_deployment_target' not specified, 'convert_to' is set to "mlprogram" and 'minimum_deployment_targer' is set to ct.target.iOS15 (which is same as ct.target.macOS12). Note: the model will not run on systems older than iOS15/macOS12/watchOS8/tvOS15. In order to make your model run on older system, please set the 'minimum_deployment_target' to iOS14/iOS13. Details please see the link: https://coremltools.readme.io/docs/unified-conversion-api#target-conversion-formats
Tuple detected at graph output. This will be flattened in the converted model.
Converting graph.
Adding op 'model.0.conv.bias' of type const
Adding op 'model.0.conv.weight' of type const
Adding op 'model.1.conv.bias' of type const
Adding op 'model.1.conv.weight' of type const
Adding op 'model.2.conv.bias' of type const
Adding op 'model.2.conv.weight' of type const
Adding op 'model.3.conv.bias' of type const
Adding op 'model.3.conv.weight' of type const
Adding op 'model.4.conv.bias' of type const
Adding op 'model.4.conv.weight' of type const
Adding op 'model.5.conv.bias' of type const
Adding op 'model.5.conv.weight' of type const
Adding op 'model.6.conv.bias' of type const
Adding op 'model.6.conv.weight' of type const
Adding op 'model.7.conv.bias' of type const
Adding op 'model.7.conv.weight' of type const
Adding op 'model.8.conv.bias' of type const
Adding op 'model.8.conv.weight' of type const
Adding op 'model.9.conv.bias' of type const
Adding op 'model.9.conv.weight' of type const
Adding op 'model.11.conv.bias' of type const
Adding op 'model.11.conv.weight' of type const
Adding op 'model.13.conv.bias' of type const
Adding op 'model.13.conv.weight' of type const
Adding op 'model.14.conv.bias' of type const
Adding op 'model.14.conv.weight' of type const
Adding op 'model.15.conv.bias' of type const
Adding op 'model.15.conv.weight' of type const
Adding op 'model.17.conv.bias' of type const
Adding op 'model.17.conv.weight' of type const
Adding op 'model.18.conv.bias' of type const
Adding op 'model.18.conv.weight' of type const
Adding op 'model.19.conv.bias' of type const
Adding op 'model.19.conv.weight' of type const
Adding op 'model.20.conv.bias' of type const
Adding op 'model.20.conv.weight' of type const
Adding op 'model.21.conv.bias' of type const
Adding op 'model.21.conv.weight' of type const
Adding op 'model.22.conv.bias' of type const
Adding op 'model.22.conv.weight' of type const
Adding op 'model.24.conv.bias' of type const
Adding op 'model.24.conv.weight' of type const
Adding op 'model.26.conv.bias' of type const
Adding op 'model.26.conv.weight' of type const
Adding op 'model.27.conv.bias' of type const
Adding op 'model.27.conv.weight' of type const
Adding op 'model.28.conv.bias' of type const
Adding op 'model.28.conv.weight' of type const
Adding op 'model.30.conv.bias' of type const
Adding op 'model.30.conv.weight' of type const
Adding op 'model.31.conv.bias' of type const
Adding op 'model.31.conv.weight' of type const
Adding op 'model.32.conv.bias' of type const
Adding op 'model.32.conv.weight' of type const
Adding op 'model.33.conv.bias' of type const
Adding op 'model.33.conv.weight' of type const
Adding op 'model.34.conv.bias' of type const
Adding op 'model.34.conv.weight' of type const
Adding op 'model.35.conv.bias' of type const
Adding op 'model.35.conv.weight' of type const
Adding op 'model.37.conv.bias' of type const
Adding op 'model.37.conv.weight' of type const
Adding op 'model.39.conv.bias' of type const
Adding op 'model.39.conv.weight' of type const
Adding op 'model.40.conv.bias' of type const
Adding op 'model.40.conv.weight' of type const
Adding op 'model.41.conv.bias' of type const
Adding op 'model.41.conv.weight' of type const
Adding op 'model.43.conv.bias' of type const
Adding op 'model.43.conv.weight' of type const
Adding op 'model.44.conv.bias' of type const
Adding op 'model.44.conv.weight' of type const
Adding op 'model.45.conv.bias' of type const
Adding op 'model.45.conv.weight' of type const
Adding op 'model.46.conv.bias' of type const
Adding op 'model.46.conv.weight' of type const
Adding op 'model.47.conv.bias' of type const
Adding op 'model.47.conv.weight' of type const
Adding op 'model.48.conv.bias' of type const
Adding op 'model.48.conv.weight' of type const
Adding op 'model.50.conv.bias' of type const
Adding op 'model.50.conv.weight' of type const
Adding op 'model.51.cv1.conv.bias' of type const
Adding op 'model.51.cv1.conv.weight' of type const
Adding op 'model.51.cv3.conv.bias' of type const
Adding op 'model.51.cv3.conv.weight' of type const
Adding op 'model.51.cv4.conv.bias' of type const
Adding op 'model.51.cv4.conv.weight' of type const
Adding op 'model.51.cv5.conv.bias' of type const
Adding op 'model.51.cv5.conv.weight' of type const
Adding op 'model.51.cv6.conv.bias' of type const
Adding op 'model.51.cv6.conv.weight' of type const
Adding op 'model.51.cv2.conv.bias' of type const
Adding op 'model.51.cv2.conv.weight' of type const
Adding op 'model.51.cv7.conv.bias' of type const
Adding op 'model.51.cv7.conv.weight' of type const
Adding op 'model.52.conv.bias' of type const
Adding op 'model.52.conv.weight' of type const
Adding op 'model.54.conv.bias' of type const
Adding op 'model.54.conv.weight' of type const
Adding op 'model.56.conv.bias' of type const
Adding op 'model.56.conv.weight' of type const
Adding op 'model.57.conv.bias' of type const
Adding op 'model.57.conv.weight' of type const
Adding op 'model.58.conv.bias' of type const
Adding op 'model.58.conv.weight' of type const
Adding op 'model.59.conv.bias' of type const
Adding op 'model.59.conv.weight' of type const
Adding op 'model.60.conv.bias' of type const
Adding op 'model.60.conv.weight' of type const
Adding op 'model.61.conv.bias' of type const
Adding op 'model.61.conv.weight' of type const
Adding op 'model.63.conv.bias' of type const
Adding op 'model.63.conv.weight' of type const
Adding op 'model.64.conv.bias' of type const
Adding op 'model.64.conv.weight' of type const
Adding op 'model.66.conv.bias' of type const
Adding op 'model.66.conv.weight' of type const
Adding op 'model.68.conv.bias' of type const
Adding op 'model.68.conv.weight' of type const
Adding op 'model.69.conv.bias' of type const
Adding op 'model.69.conv.weight' of type const
Adding op 'model.70.conv.bias' of type const
Adding op 'model.70.conv.weight' of type const
Adding op 'model.71.conv.bias' of type const
Adding op 'model.71.conv.weight' of type const
Adding op 'model.72.conv.bias' of type const
Adding op 'model.72.conv.weight' of type const
Adding op 'model.73.conv.bias' of type const
Adding op 'model.73.conv.weight' of type const
Adding op 'model.75.conv.bias' of type const
Adding op 'model.75.conv.weight' of type const
Adding op 'model.77.conv.bias' of type const
Adding op 'model.77.conv.weight' of type const
Adding op 'model.78.conv.bias' of type const
Adding op 'model.78.conv.weight' of type const
Adding op 'model.79.conv.bias' of type const
Adding op 'model.79.conv.weight' of type const
Adding op 'model.81.conv.bias' of type const
Adding op 'model.81.conv.weight' of type const
Adding op 'model.82.conv.bias' of type const
Adding op 'model.82.conv.weight' of type const
Adding op 'model.83.conv.bias' of type const
Adding op 'model.83.conv.weight' of type const
Adding op 'model.84.conv.bias' of type const
Adding op 'model.84.conv.weight' of type const
Adding op 'model.85.conv.bias' of type const
Adding op 'model.85.conv.weight' of type const
Adding op 'model.86.conv.bias' of type const
Adding op 'model.86.conv.weight' of type const
Adding op 'model.88.conv.bias' of type const
Adding op 'model.88.conv.weight' of type const
Adding op 'model.90.conv.bias' of type const
Adding op 'model.90.conv.weight' of type const
Adding op 'model.91.conv.bias' of type const
Adding op 'model.91.conv.weight' of type const
Adding op 'model.92.conv.bias' of type const
Adding op 'model.92.conv.weight' of type const
Adding op 'model.94.conv.bias' of type const
Adding op 'model.94.conv.weight' of type const
Adding op 'model.95.conv.bias' of type const
Adding op 'model.95.conv.weight' of type const
Adding op 'model.96.conv.bias' of type const
Adding op 'model.96.conv.weight' of type const
Adding op 'model.97.conv.bias' of type const
Adding op 'model.97.conv.weight' of type const
Adding op 'model.98.conv.bias' of type const
Adding op 'model.98.conv.weight' of type const
Adding op 'model.99.conv.bias' of type const
Adding op 'model.99.conv.weight' of type const
Adding op 'model.101.conv.bias' of type const
Adding op 'model.101.conv.weight' of type const
Adding op 'model.102.rbr_reparam.bias' of type const
Adding op 'model.102.rbr_reparam.weight' of type const
Adding op 'model.103.rbr_reparam.bias' of type const
Adding op 'model.103.rbr_reparam.weight' of type const
Adding op 'model.104.rbr_reparam.bias' of type const
Adding op 'model.104.rbr_reparam.weight' of type const
Adding op 'model.105.anchor_grid' of type const
Adding op 'model.105.m.0.bias' of type const
Adding op 'model.105.m.0.weight' of type const
Adding op 'model.105.m.1.bias' of type const
Adding op 'model.105.m.1.weight' of type const
Adding op 'model.105.m.2.bias' of type const
Adding op 'model.105.m.2.weight' of type const
Converting PyTorch Frontend ==> MIL Ops:   0%|                                              | 0/1203 [00:00<?, ? ops/s]Converting op 170 : constant
Adding op '170' of type const
Converting op 171 : constant
Adding op '171' of type const
Converting op 172 : constant
Adding op '172' of type const
Converting op 173 : constant
Adding op '173' of type const
Converting op 177 : listconstruct
Adding op '177' of type const
Converting op 178 : listconstruct
Adding op '178' of type const
Converting op 179 : listconstruct
Adding op '179' of type const
Converting op 180 : listconstruct
Adding op '180' of type const
Converting op x.3 : _convolution
Adding op 'x.3' of type conv
Adding op 'x.3_pad_type_0' of type const
Adding op 'x.3_pad_0' of type const
Converting op 182 : sigmoid
Adding op '182' of type sigmoid
Converting op input.1 : mul
Adding op 'input.1' of type mul
Converting op 184 : constant
Adding op '184' of type const
Converting op 185 : constant
Adding op '185' of type const
Converting op 186 : constant
Adding op '186' of type const
Converting op 187 : constant
Adding op '187' of type const
Converting op 188 : constant
Adding op '188' of type const
Converting op 192 : listconstruct
Adding op '192' of type const
Converting op 193 : listconstruct
Adding op '193' of type const
Converting op 194 : listconstruct
Adding op '194' of type const
Converting op 195 : listconstruct
Adding op '195' of type const
Converting op x.5 : _convolution
Adding op 'x.5' of type conv
Adding op 'x.5_pad_type_0' of type const
Adding op 'x.5_pad_0' of type const
Converting op 197 : sigmoid
Adding op '197' of type sigmoid
Converting op input.3 : mul
Adding op 'input.3' of type mul
Converting op 199 : constant
Adding op '199' of type const
Converting op 200 : constant
Adding op '200' of type const
Converting op 201 : constant
Adding op '201' of type const
Converting op 202 : constant
Adding op '202' of type const
Converting op 206 : listconstruct
Adding op '206' of type const
Converting op 207 : listconstruct
Adding op '207' of type const
Converting op 208 : listconstruct
Adding op '208' of type const
Converting op 209 : listconstruct
Adding op '209' of type const
Converting op x.7 : _convolution
Adding op 'x.7' of type conv
Adding op 'x.7_pad_type_0' of type const
Adding op 'x.7_pad_0' of type const
Converting op 211 : sigmoid
Adding op '211' of type sigmoid
Converting op input.5 : mul
Adding op 'input.5' of type mul
Converting op 213 : constant
Adding op '213' of type const
Converting op 214 : constant
Adding op '214' of type const
Converting op 215 : constant
Adding op '215' of type const
Converting op 216 : constant
Adding op '216' of type const
Converting op 217 : constant
Adding op '217' of type const
Converting op 221 : listconstruct
Adding op '221' of type const
Converting op 222 : listconstruct
Adding op '222' of type const
Converting op 223 : listconstruct
Adding op '223' of type const
Converting op 224 : listconstruct
Adding op '224' of type const
Converting op x.9 : _convolution
Adding op 'x.9' of type conv
Adding op 'x.9_pad_type_0' of type const
Adding op 'x.9_pad_0' of type const
Converting op 226 : sigmoid
Adding op '226' of type sigmoid
Converting op input.7 : mul
Adding op 'input.7' of type mul
Converting op 228 : constant
Adding op '228' of type const
Converting op 229 : constant
Adding op '229' of type const
Converting op 230 : constant
Adding op '230' of type const
Converting op 231 : constant
Adding op '231' of type const
Converting op 235 : listconstruct
Adding op '235' of type const
Converting op 236 : listconstruct
Adding op '236' of type const
Converting op 237 : listconstruct
Adding op '237' of type const
Converting op 238 : listconstruct
Adding op '238' of type const
Converting op x.11 : _convolution
Adding op 'x.11' of type conv
Adding op 'x.11_pad_type_0' of type const
Adding op 'x.11_pad_0' of type const
Converting op 240 : sigmoid
Adding op '240' of type sigmoid
Converting op 241 : mul
Adding op '241' of type mul
Converting op 242 : constant
Adding op '242' of type const
Converting op 243 : constant
Adding op '243' of type const
Converting op 244 : constant
Adding op '244' of type const
Converting op 245 : constant
Adding op '245' of type const
Converting op 249 : listconstruct
Adding op '249' of type const
Converting op 250 : listconstruct
Adding op '250' of type const
Converting op 251 : listconstruct
Adding op '251' of type const
Converting op 252 : listconstruct
Adding op '252' of type const
Converting op x.13 : _convolution
Adding op 'x.13' of type conv
Adding op 'x.13_pad_type_0' of type const
Adding op 'x.13_pad_0' of type const
Converting op 254 : sigmoid
Adding op '254' of type sigmoid
Converting op input.9 : mul
Adding op 'input.9' of type mul
Converting op 256 : constant
Adding op '256' of type const
Converting op 257 : constant
Adding op '257' of type const
Converting op 258 : constant
Adding op '258' of type const
Converting op 259 : constant
Adding op '259' of type const
Converting op 263 : listconstruct
Adding op '263' of type const
Converting op 264 : listconstruct
Adding op '264' of type const
Converting op 265 : listconstruct
Adding op '265' of type const
Converting op 266 : listconstruct
Adding op '266' of type const
Converting op x.15 : _convolution
Adding op 'x.15' of type conv
Adding op 'x.15_pad_type_0' of type const
Adding op 'x.15_pad_0' of type const
Converting op 268 : sigmoid
Adding op '268' of type sigmoid
Converting op input.11 : mul
Adding op 'input.11' of type mul
Converting op 270 : constant
Adding op '270' of type const
Converting op 271 : constant
Adding op '271' of type const
Converting op 272 : constant
Adding op '272' of type const
Converting op 273 : constant
Adding op '273' of type const
Converting op 277 : listconstruct
Adding op '277' of type const
Converting op 278 : listconstruct
Adding op '278' of type const
Converting op 279 : listconstruct
Adding op '279' of type const
Converting op 280 : listconstruct
Adding op '280' of type const
Converting op x.17 : _convolution
Adding op 'x.17' of type conv
Adding op 'x.17_pad_type_0' of type const
Adding op 'x.17_pad_0' of type const
Converting op 282 : sigmoid
Adding op '282' of type sigmoid
Converting op input.13 : mul
Adding op 'input.13' of type mul
Converting op 284 : constant
Adding op '284' of type const
Converting op 285 : constant
Adding op '285' of type const
Converting op 286 : constant
Adding op '286' of type const
Converting op 287 : constant
Adding op '287' of type const
Converting op 291 : listconstruct
Adding op '291' of type const
Converting op 292 : listconstruct
Adding op '292' of type const
Converting op 293 : listconstruct
Adding op '293' of type const
Converting op 294 : listconstruct
Adding op '294' of type const
Converting op x.19 : _convolution
Adding op 'x.19' of type conv
Adding op 'x.19_pad_type_0' of type const
Adding op 'x.19_pad_0' of type const
Converting op 296 : sigmoid
Adding op '296' of type sigmoid
Converting op input.15 : mul
Adding op 'input.15' of type mul
Converting op 298 : constant
Adding op '298' of type const
Converting op 299 : constant
Adding op '299' of type const
Converting op 300 : constant
Adding op '300' of type const
Converting op 301 : constant
Adding op '301' of type const
Converting op 305 : listconstruct
Adding op '305' of type const
Converting op 306 : listconstruct
Adding op '306' of type const
Converting op 307 : listconstruct
Adding op '307' of type const
Converting op 308 : listconstruct
Adding op '308' of type const
Converting op x.21 : _convolution
Adding op 'x.21' of type conv
Adding op 'x.21_pad_type_0' of type const
Adding op 'x.21_pad_0' of type const
Converting op 310 : sigmoid
Adding op '310' of type sigmoid
Converting op 311 : mul
Adding op '311' of type mul
Converting op 312 : constant
Adding op '312' of type const
Converting op 313 : listconstruct
Converting op input.17 : cat
Adding op 'input.17' of type concat
Adding op 'input.17_interleave_0' of type const
Converting op 315 : constant
Adding op '315' of type const
Converting op 316 : constant
Adding op '316' of type const
Converting op 317 : constant
Adding op '317' of type const
Converting op 318 : constant
Adding op '318' of type const
Converting op 322 : listconstruct
Adding op '322' of type const
Converting op 323 : listconstruct
Adding op '323' of type const
Converting op 324 : listconstruct
Adding op '324' of type const
Converting op 325 : listconstruct
Adding op '325' of type const
Converting op x.23 : _convolution
Adding op 'x.23' of type conv
Adding op 'x.23_pad_type_0' of type const
Adding op 'x.23_pad_0' of type const
Converting op 327 : sigmoid
Adding op '327' of type sigmoid
Converting op input.19 : mul
Adding op 'input.19' of type mul
Converting op 329 : constant
Adding op '329' of type const
Converting op 330 : constant
Adding op '330' of type const
Converting op 331 : constant
Adding op '331' of type const
Converting op 332 : constant
Adding op '332' of type const
Converting op 333 : listconstruct
Adding op '333' of type const
Converting op 334 : listconstruct
Adding op '334' of type const
Converting op 335 : listconstruct
Adding op '335' of type const
Converting op 336 : listconstruct
Adding op '336' of type const
Converting op input.21 : max_pool2d
Adding op 'input.21' of type max_pool
Adding op 'input.21_pad_type_0' of type const
Adding op 'input.21_pad_0' of type const
Adding op 'input.21_ceil_mode_0' of type const
Converting op 338 : constant
Adding op '338' of type const
Converting op 339 : constant
Adding op '339' of type const
Converting op 340 : constant
Adding op '340' of type const
Converting op 341 : constant
Adding op '341' of type const
Converting op 345 : listconstruct
Adding op '345' of type const
Converting op 346 : listconstruct
Adding op '346' of type const
Converting op 347 : listconstruct
Adding op '347' of type const
Converting op 348 : listconstruct
Adding op '348' of type const
Converting op x.25 : _convolution
Adding op 'x.25' of type conv
Adding op 'x.25_pad_type_0' of type const
Adding op 'x.25_pad_0' of type const
Converting op 350 : sigmoid
Adding op '350' of type sigmoid
Converting op 351 : mul
Adding op '351' of type mul
Converting op 352 : constant
Adding op '352' of type const
Converting op 353 : constant
Adding op '353' of type const
Converting op 354 : constant
Adding op '354' of type const
Converting op 355 : constant
Adding op '355' of type const
Converting op 359 : listconstruct
Adding op '359' of type const
Converting op 360 : listconstruct
Adding op '360' of type const
Converting op 361 : listconstruct
Adding op '361' of type const
Converting op 362 : listconstruct
Adding op '362' of type const
Converting op x.27 : _convolution
Adding op 'x.27' of type conv
Adding op 'x.27_pad_type_0' of type const
Adding op 'x.27_pad_0' of type const
Converting op 364 : sigmoid
Adding op '364' of type sigmoid
Converting op input.23 : mul
Adding op 'input.23' of type mul
Converting op 366 : constant
Adding op '366' of type const
Converting op 367 : constant
Adding op '367' of type const
Converting op 368 : constant
Adding op '368' of type const
Converting op 369 : constant
Adding op '369' of type const
Converting op 370 : constant
Adding op '370' of type const
Converting op 374 : listconstruct
Adding op '374' of type const
Converting op 375 : listconstruct
Adding op '375' of type const
Converting op 376 : listconstruct
Adding op '376' of type const
Converting op 377 : listconstruct
Adding op '377' of type const
Converting op x.29 : _convolution
Adding op 'x.29' of type conv
Adding op 'x.29_pad_type_0' of type const
Adding op 'x.29_pad_0' of type const
Converting op 379 : sigmoid
Adding op '379' of type sigmoid
Converting op 380 : mul
Adding op '380' of type mul
Converting op 381 : constant
Adding op '381' of type const
Converting op 382 : listconstruct
Converting op input.25 : cat
Adding op 'input.25' of type concat
Adding op 'input.25_interleave_0' of type const
Converting op 384 : constant
Adding op '384' of type const
Converting op 385 : constant
Adding op '385' of type const
Converting op 386 : constant
Adding op '386' of type const
Converting op 387 : constant
Adding op '387' of type const
Converting op 391 : listconstruct
Adding op '391' of type const
Converting op 392 : listconstruct
Adding op '392' of type const
Converting op 393 : listconstruct
Adding op '393' of type const
Converting op 394 : listconstruct
Adding op '394' of type const
Converting op x.31 : _convolution
Adding op 'x.31' of type conv
Adding op 'x.31_pad_type_0' of type const
Adding op 'x.31_pad_0' of type const
Converting op 396 : sigmoid
Adding op '396' of type sigmoid
Converting op 397 : mul
Adding op '397' of type mul
Converting op 398 : constant
Adding op '398' of type const
Converting op 399 : constant
Adding op '399' of type const
Converting op 400 : constant
Adding op '400' of type const
Converting op 401 : constant
Adding op '401' of type const
Converting op 405 : listconstruct
Adding op '405' of type const
Converting op 406 : listconstruct
Adding op '406' of type const
Converting op 407 : listconstruct
Adding op '407' of type const
Converting op 408 : listconstruct
Adding op '408' of type const
Converting op x.33 : _convolution
Adding op 'x.33' of type conv
Adding op 'x.33_pad_type_0' of type const
Adding op 'x.33_pad_0' of type const
Converting op 410 : sigmoid
Adding op '410' of type sigmoid
Converting op input.27 : mul
Adding op 'input.27' of type mul
Converting op 412 : constant
Adding op '412' of type const
Converting op 413 : constant
Adding op '413' of type const
Converting op 414 : constant
Adding op '414' of type const
Converting op 415 : constant
Adding op '415' of type const
Converting op 419 : listconstruct
Adding op '419' of type const
Converting op 420 : listconstruct
Adding op '420' of type const
Converting op 421 : listconstruct
Adding op '421' of type const
Converting op 422 : listconstruct
Adding op '422' of type const
Converting op x.35 : _convolution
Adding op 'x.35' of type conv
Adding op 'x.35_pad_type_0' of type const
Adding op 'x.35_pad_0' of type const
Converting op 424 : sigmoid
Adding op '424' of type sigmoid
Converting op input.29 : mul
Adding op 'input.29' of type mul
Converting op 426 : constant
Adding op '426' of type const
Converting op 427 : constant
Adding op '427' of type const
Converting op 428 : constant
Adding op '428' of type const
Converting op 429 : constant
Adding op '429' of type const
Converting op 433 : listconstruct
Adding op '433' of type const
Converting op 434 : listconstruct
Adding op '434' of type const
Converting op 435 : listconstruct
Adding op '435' of type const
Converting op 436 : listconstruct
Adding op '436' of type const
Converting op x.37 : _convolution
Adding op 'x.37' of type conv
Adding op 'x.37_pad_type_0' of type const
Adding op 'x.37_pad_0' of type const
Converting op 438 : sigmoid
Adding op '438' of type sigmoid
Converting op input.31 : mul
Adding op 'input.31' of type mul
Converting op 440 : constant
Adding op '440' of type const
Converting op 441 : constant
Adding op '441' of type const
Converting op 442 : constant
Adding op '442' of type const
Converting op 443 : constant
Adding op '443' of type const
Converting op 447 : listconstruct
Adding op '447' of type const
Converting op 448 : listconstruct
Adding op '448' of type const
Converting op 449 : listconstruct
Adding op '449' of type const
Converting op 450 : listconstruct
Adding op '450' of type const
Converting op x.39 : _convolution
Adding op 'x.39' of type conv
Adding op 'x.39_pad_type_0' of type const
Adding op 'x.39_pad_0' of type const
Converting op 452 : sigmoid
Adding op '452' of type sigmoid
Converting op input.33 : mul
Adding op 'input.33' of type mul
Converting op 454 : constant
Adding op '454' of type const
Converting op 455 : constant
Adding op '455' of type const
Converting op 456 : constant
Adding op '456' of type const
Converting op 457 : constant
Adding op '457' of type const
Converting op 461 : listconstruct
Adding op '461' of type const
Converting op 462 : listconstruct
Adding op '462' of type const
Converting op 463 : listconstruct
Adding op '463' of type const
Converting PyTorch Frontend ==> MIL Ops:  19%|██████▌                           | 234/1203 [00:00<00:00, 2328.07 ops/s]Converting op 464 : listconstruct
Adding op '464' of type const
Converting op x.41 : _convolution
Adding op 'x.41' of type conv
Adding op 'x.41_pad_type_0' of type const
Adding op 'x.41_pad_0' of type const
Converting op 466 : sigmoid
Adding op '466' of type sigmoid
Converting op 467 : mul
Adding op '467' of type mul
Converting op 468 : constant
Adding op '468' of type const
Converting op 469 : listconstruct
Converting op input.35 : cat
Adding op 'input.35' of type concat
Adding op 'input.35_interleave_0' of type const
Converting op 471 : constant
Adding op '471' of type const
Converting op 472 : constant
Adding op '472' of type const
Converting op 473 : constant
Adding op '473' of type const
Converting op 474 : constant
Adding op '474' of type const
Converting op 478 : listconstruct
Adding op '478' of type const
Converting op 479 : listconstruct
Adding op '479' of type const
Converting op 480 : listconstruct
Adding op '480' of type const
Converting op 481 : listconstruct
Adding op '481' of type const
Converting op x.43 : _convolution
Adding op 'x.43' of type conv
Adding op 'x.43_pad_type_0' of type const
Adding op 'x.43_pad_0' of type const
Converting op 483 : sigmoid
Adding op '483' of type sigmoid
Converting op input.37 : mul
Adding op 'input.37' of type mul
Converting op 485 : constant
Adding op '485' of type const
Converting op 486 : constant
Adding op '486' of type const
Converting op 487 : constant
Adding op '487' of type const
Converting op 488 : constant
Adding op '488' of type const
Converting op 489 : listconstruct
Adding op '489' of type const
Converting op 490 : listconstruct
Adding op '490' of type const
Converting op 491 : listconstruct
Adding op '491' of type const
Converting op 492 : listconstruct
Adding op '492' of type const
Converting op input.39 : max_pool2d
Adding op 'input.39' of type max_pool
Adding op 'input.39_pad_type_0' of type const
Adding op 'input.39_pad_0' of type const
Adding op 'input.39_ceil_mode_0' of type const
Converting op 494 : constant
Adding op '494' of type const
Converting op 495 : constant
Adding op '495' of type const
Converting op 496 : constant
Adding op '496' of type const
Converting op 497 : constant
Adding op '497' of type const
Converting op 501 : listconstruct
Adding op '501' of type const
Converting op 502 : listconstruct
Adding op '502' of type const
Converting op 503 : listconstruct
Adding op '503' of type const
Converting op 504 : listconstruct
Adding op '504' of type const
Converting op x.45 : _convolution
Adding op 'x.45' of type conv
Adding op 'x.45_pad_type_0' of type const
Adding op 'x.45_pad_0' of type const
Converting op 506 : sigmoid
Adding op '506' of type sigmoid
Converting op 507 : mul
Adding op '507' of type mul
Converting op 508 : constant
Adding op '508' of type const
Converting op 509 : constant
Adding op '509' of type const
Converting op 510 : constant
Adding op '510' of type const
Converting op 511 : constant
Adding op '511' of type const
Converting op 515 : listconstruct
Adding op '515' of type const
Converting op 516 : listconstruct
Adding op '516' of type const
Converting op 517 : listconstruct
Adding op '517' of type const
Converting op 518 : listconstruct
Adding op '518' of type const
Converting op x.47 : _convolution
Adding op 'x.47' of type conv
Adding op 'x.47_pad_type_0' of type const
Adding op 'x.47_pad_0' of type const
Converting op 520 : sigmoid
Adding op '520' of type sigmoid
Converting op input.41 : mul
Adding op 'input.41' of type mul
Converting op 522 : constant
Adding op '522' of type const
Converting op 523 : constant
Adding op '523' of type const
Converting op 524 : constant
Adding op '524' of type const
Converting op 525 : constant
Adding op '525' of type const
Converting op 526 : constant
Adding op '526' of type const
Converting op 530 : listconstruct
Adding op '530' of type const
Converting op 531 : listconstruct
Adding op '531' of type const
Converting op 532 : listconstruct
Adding op '532' of type const
Converting op 533 : listconstruct
Adding op '533' of type const
Converting op x.49 : _convolution
Adding op 'x.49' of type conv
Adding op 'x.49_pad_type_0' of type const
Adding op 'x.49_pad_0' of type const
Converting op 535 : sigmoid
Adding op '535' of type sigmoid
Converting op 536 : mul
Adding op '536' of type mul
Converting op 537 : constant
Adding op '537' of type const
Converting op 538 : listconstruct
Converting op input.43 : cat
Adding op 'input.43' of type concat
Adding op 'input.43_interleave_0' of type const
Converting op 540 : constant
Adding op '540' of type const
Converting op 541 : constant
Adding op '541' of type const
Converting op 542 : constant
Adding op '542' of type const
Converting op 543 : constant
Adding op '543' of type const
Converting op 547 : listconstruct
Adding op '547' of type const
Converting op 548 : listconstruct
Adding op '548' of type const
Converting op 549 : listconstruct
Adding op '549' of type const
Converting op 550 : listconstruct
Adding op '550' of type const
Converting op x.51 : _convolution
Adding op 'x.51' of type conv
Adding op 'x.51_pad_type_0' of type const
Adding op 'x.51_pad_0' of type const
Converting op 552 : sigmoid
Adding op '552' of type sigmoid
Converting op 553 : mul
Adding op '553' of type mul
Converting op 554 : constant
Adding op '554' of type const
Converting op 555 : constant
Adding op '555' of type const
Converting op 556 : constant
Adding op '556' of type const
Converting op 557 : constant
Adding op '557' of type const
Converting op 561 : listconstruct
Adding op '561' of type const
Converting op 562 : listconstruct
Adding op '562' of type const
Converting op 563 : listconstruct
Adding op '563' of type const
Converting op 564 : listconstruct
Adding op '564' of type const
Converting op x.53 : _convolution
Adding op 'x.53' of type conv
Adding op 'x.53_pad_type_0' of type const
Adding op 'x.53_pad_0' of type const
Converting op 566 : sigmoid
Adding op '566' of type sigmoid
Converting op input.45 : mul
Adding op 'input.45' of type mul
Converting op 568 : constant
Adding op '568' of type const
Converting op 569 : constant
Adding op '569' of type const
Converting op 570 : constant
Adding op '570' of type const
Converting op 571 : constant
Adding op '571' of type const
Converting op 575 : listconstruct
Adding op '575' of type const
Converting op 576 : listconstruct
Adding op '576' of type const
Converting op 577 : listconstruct
Adding op '577' of type const
Converting op 578 : listconstruct
Adding op '578' of type const
Converting op x.55 : _convolution
Adding op 'x.55' of type conv
Adding op 'x.55_pad_type_0' of type const
Adding op 'x.55_pad_0' of type const
Converting op 580 : sigmoid
Adding op '580' of type sigmoid
Converting op input.47 : mul
Adding op 'input.47' of type mul
Converting op 582 : constant
Adding op '582' of type const
Converting op 583 : constant
Adding op '583' of type const
Converting op 584 : constant
Adding op '584' of type const
Converting op 585 : constant
Adding op '585' of type const
Converting op 589 : listconstruct
Adding op '589' of type const
Converting op 590 : listconstruct
Adding op '590' of type const
Converting op 591 : listconstruct
Adding op '591' of type const
Converting op 592 : listconstruct
Adding op '592' of type const
Converting op x.57 : _convolution
Adding op 'x.57' of type conv
Adding op 'x.57_pad_type_0' of type const
Adding op 'x.57_pad_0' of type const
Converting op 594 : sigmoid
Adding op '594' of type sigmoid
Converting op input.49 : mul
Adding op 'input.49' of type mul
Converting op 596 : constant
Adding op '596' of type const
Converting op 597 : constant
Adding op '597' of type const
Converting op 598 : constant
Adding op '598' of type const
Converting op 599 : constant
Adding op '599' of type const
Converting op 603 : listconstruct
Adding op '603' of type const
Converting op 604 : listconstruct
Adding op '604' of type const
Converting op 605 : listconstruct
Adding op '605' of type const
Converting op 606 : listconstruct
Adding op '606' of type const
Converting op x.59 : _convolution
Adding op 'x.59' of type conv
Adding op 'x.59_pad_type_0' of type const
Adding op 'x.59_pad_0' of type const
Converting op 608 : sigmoid
Adding op '608' of type sigmoid
Converting op input.51 : mul
Adding op 'input.51' of type mul
Converting op 610 : constant
Adding op '610' of type const
Converting op 611 : constant
Adding op '611' of type const
Converting op 612 : constant
Adding op '612' of type const
Converting op 613 : constant
Adding op '613' of type const
Converting op 617 : listconstruct
Adding op '617' of type const
Converting op 618 : listconstruct
Adding op '618' of type const
Converting op 619 : listconstruct
Adding op '619' of type const
Converting op 620 : listconstruct
Adding op '620' of type const
Converting op x.61 : _convolution
Adding op 'x.61' of type conv
Adding op 'x.61_pad_type_0' of type const
Adding op 'x.61_pad_0' of type const
Converting op 622 : sigmoid
Adding op '622' of type sigmoid
Converting op 623 : mul
Adding op '623' of type mul
Converting op 624 : constant
Adding op '624' of type const
Converting op 625 : listconstruct
Converting op input.53 : cat
Adding op 'input.53' of type concat
Adding op 'input.53_interleave_0' of type const
Converting op 627 : constant
Adding op '627' of type const
Converting op 628 : constant
Adding op '628' of type const
Converting op 629 : constant
Adding op '629' of type const
Converting op 630 : constant
Adding op '630' of type const
Converting op 634 : listconstruct
Adding op '634' of type const
Converting op 635 : listconstruct
Adding op '635' of type const
Converting op 636 : listconstruct
Adding op '636' of type const
Converting op 637 : listconstruct
Adding op '637' of type const
Converting op x.63 : _convolution
Adding op 'x.63' of type conv
Adding op 'x.63_pad_type_0' of type const
Adding op 'x.63_pad_0' of type const
Converting op 639 : sigmoid
Adding op '639' of type sigmoid
Converting op input.55 : mul
Adding op 'input.55' of type mul
Converting op 641 : constant
Adding op '641' of type const
Converting op 642 : constant
Adding op '642' of type const
Converting op 643 : constant
Adding op '643' of type const
Converting op 644 : constant
Adding op '644' of type const
Converting op 645 : listconstruct
Adding op '645' of type const
Converting op 646 : listconstruct
Adding op '646' of type const
Converting op 647 : listconstruct
Adding op '647' of type const
Converting op 648 : listconstruct
Adding op '648' of type const
Converting op input.57 : max_pool2d
Adding op 'input.57' of type max_pool
Adding op 'input.57_pad_type_0' of type const
Adding op 'input.57_pad_0' of type const
Adding op 'input.57_ceil_mode_0' of type const
Converting op 650 : constant
Adding op '650' of type const
Converting op 651 : constant
Adding op '651' of type const
Converting op 652 : constant
Adding op '652' of type const
Converting op 653 : constant
Adding op '653' of type const
Converting op 657 : listconstruct
Adding op '657' of type const
Converting op 658 : listconstruct
Adding op '658' of type const
Converting op 659 : listconstruct
Adding op '659' of type const
Converting op 660 : listconstruct
Adding op '660' of type const
Converting op x.65 : _convolution
Adding op 'x.65' of type conv
Adding op 'x.65_pad_type_0' of type const
Adding op 'x.65_pad_0' of type const
Converting op 662 : sigmoid
Adding op '662' of type sigmoid
Converting op 663 : mul
Adding op '663' of type mul
Converting op 664 : constant
Adding op '664' of type const
Converting op 665 : constant
Adding op '665' of type const
Converting op 666 : constant
Adding op '666' of type const
Converting op 667 : constant
Adding op '667' of type const
Converting op 671 : listconstruct
Adding op '671' of type const
Converting op 672 : listconstruct
Adding op '672' of type const
Converting op 673 : listconstruct
Adding op '673' of type const
Converting op 674 : listconstruct
Adding op '674' of type const
Converting op x.67 : _convolution
Adding op 'x.67' of type conv
Adding op 'x.67_pad_type_0' of type const
Adding op 'x.67_pad_0' of type const
Converting op 676 : sigmoid
Adding op '676' of type sigmoid
Converting op input.59 : mul
Adding op 'input.59' of type mul
Converting op 678 : constant
Adding op '678' of type const
Converting op 679 : constant
Adding op '679' of type const
Converting op 680 : constant
Adding op '680' of type const
Converting op 681 : constant
Adding op '681' of type const
Converting op 682 : constant
Adding op '682' of type const
Converting op 686 : listconstruct
Adding op '686' of type const
Converting op 687 : listconstruct
Adding op '687' of type const
Converting op 688 : listconstruct
Adding op '688' of type const
Converting op 689 : listconstruct
Adding op '689' of type const
Converting op x.69 : _convolution
Adding op 'x.69' of type conv
Adding op 'x.69_pad_type_0' of type const
Adding op 'x.69_pad_0' of type const
Converting op 691 : sigmoid
Adding op '691' of type sigmoid
Converting op 692 : mul
Adding op '692' of type mul
Converting op 693 : constant
Adding op '693' of type const
Converting op 694 : listconstruct
Converting op input.61 : cat
Adding op 'input.61' of type concat
Adding op 'input.61_interleave_0' of type const
Converting op 696 : constant
Adding op '696' of type const
Converting op 697 : constant
Adding op '697' of type const
Converting op 698 : constant
Adding op '698' of type const
Converting op 699 : constant
Adding op '699' of type const
Converting op 703 : listconstruct
Adding op '703' of type const
Converting op 704 : listconstruct
Adding op '704' of type const
Converting op 705 : listconstruct
Adding op '705' of type const
Converting op 706 : listconstruct
Adding op '706' of type const
Converting op x.71 : _convolution
Adding op 'x.71' of type conv
Adding op 'x.71_pad_type_0' of type const
Adding op 'x.71_pad_0' of type const
Converting op 708 : sigmoid
Adding op '708' of type sigmoid
Converting op 709 : mul
Adding op '709' of type mul
Converting op 710 : constant
Adding op '710' of type const
Converting op 711 : constant
Adding op '711' of type const
Converting op 712 : constant
Adding op '712' of type const
Converting op 713 : constant
Adding op '713' of type const
Converting op 717 : listconstruct
Adding op '717' of type const
Converting op 718 : listconstruct
Adding op '718' of type const
Converting op 719 : listconstruct
Adding op '719' of type const
Converting op 720 : listconstruct
Adding op '720' of type const
Converting op x.73 : _convolution
Adding op 'x.73' of type conv
Adding op 'x.73_pad_type_0' of type const
Adding op 'x.73_pad_0' of type const
Converting op 722 : sigmoid
Adding op '722' of type sigmoid
Converting op input.63 : mul
Adding op 'input.63' of type mul
Converting op 724 : constant
Adding op '724' of type const
Converting op 725 : constant
Adding op '725' of type const
Converting op 726 : constant
Adding op '726' of type const
Converting op 727 : constant
Adding op '727' of type const
Converting op 731 : listconstruct
Adding op '731' of type const
Converting op 732 : listconstruct
Adding op '732' of type const
Converting op 733 : listconstruct
Adding op '733' of type const
Converting op 734 : listconstruct
Adding op '734' of type const
Converting op x.75 : _convolution
Adding op 'x.75' of type conv
Adding op 'x.75_pad_type_0' of type const
Adding op 'x.75_pad_0' of type const
Converting op 736 : sigmoid
Adding op '736' of type sigmoid
Converting op input.65 : mul
Adding op 'input.65' of type mul
Converting op 738 : constant
Adding op '738' of type const
Converting op 739 : constant
Adding op '739' of type const
Converting op 740 : constant
Adding op '740' of type const
Converting op 741 : constant
Adding op '741' of type const
Converting op 745 : listconstruct
Adding op '745' of type const
Converting op 746 : listconstruct
Adding op '746' of type const
Converting op 747 : listconstruct
Adding op '747' of type const
Converting op 748 : listconstruct
Adding op '748' of type const
Converting op x.77 : _convolution
Adding op 'x.77' of type conv
Adding op 'x.77_pad_type_0' of type const
Adding op 'x.77_pad_0' of type const
Converting op 750 : sigmoid
Adding op '750' of type sigmoid
Converting PyTorch Frontend ==> MIL Ops:  39%|█████████████▏                    | 467/1203 [00:00<00:00, 2274.08 ops/s]Converting op input.67 : mul
Adding op 'input.67' of type mul
Converting op 752 : constant
Adding op '752' of type const
Converting op 753 : constant
Adding op '753' of type const
Converting op 754 : constant
Adding op '754' of type const
Converting op 755 : constant
Adding op '755' of type const
Converting op 759 : listconstruct
Adding op '759' of type const
Converting op 760 : listconstruct
Adding op '760' of type const
Converting op 761 : listconstruct
Adding op '761' of type const
Converting op 762 : listconstruct
Adding op '762' of type const
Converting op x.79 : _convolution
Adding op 'x.79' of type conv
Adding op 'x.79_pad_type_0' of type const
Adding op 'x.79_pad_0' of type const
Converting op 764 : sigmoid
Adding op '764' of type sigmoid
Converting op input.69 : mul
Adding op 'input.69' of type mul
Converting op 766 : constant
Adding op '766' of type const
Converting op 767 : constant
Adding op '767' of type const
Converting op 768 : constant
Adding op '768' of type const
Converting op 769 : constant
Adding op '769' of type const
Converting op 773 : listconstruct
Adding op '773' of type const
Converting op 774 : listconstruct
Adding op '774' of type const
Converting op 775 : listconstruct
Adding op '775' of type const
Converting op 776 : listconstruct
Adding op '776' of type const
Converting op x.81 : _convolution
Adding op 'x.81' of type conv
Adding op 'x.81_pad_type_0' of type const
Adding op 'x.81_pad_0' of type const
Converting op 778 : sigmoid
Adding op '778' of type sigmoid
Converting op 779 : mul
Adding op '779' of type mul
Converting op 780 : constant
Adding op '780' of type const
Converting op 781 : listconstruct
Converting op input.71 : cat
Adding op 'input.71' of type concat
Adding op 'input.71_interleave_0' of type const
Converting op 783 : constant
Adding op '783' of type const
Converting op 784 : constant
Adding op '784' of type const
Converting op 785 : constant
Adding op '785' of type const
Converting op 786 : constant
Adding op '786' of type const
Converting op 790 : listconstruct
Adding op '790' of type const
Converting op 791 : listconstruct
Adding op '791' of type const
Converting op 792 : listconstruct
Adding op '792' of type const
Converting op 793 : listconstruct
Adding op '793' of type const
Converting op x.83 : _convolution
Adding op 'x.83' of type conv
Adding op 'x.83_pad_type_0' of type const
Adding op 'x.83_pad_0' of type const
Converting op 795 : sigmoid
Adding op '795' of type sigmoid
Converting op input.73 : mul
Adding op 'input.73' of type mul
Converting op 797 : constant
Adding op '797' of type const
Converting op 798 : constant
Adding op '798' of type const
Converting op 799 : constant
Adding op '799' of type const
Converting op 800 : constant
Adding op '800' of type const
Converting op 801 : constant
Adding op '801' of type const
Converting op 802 : constant
Adding op '802' of type const
Converting op 803 : constant
Adding op '803' of type const
Converting op 804 : constant
Adding op '804' of type const
Converting op 805 : constant
Adding op '805' of type const
Converting op 806 : constant
Adding op '806' of type const
Converting op 817 : listconstruct
Adding op '817' of type const
Converting op 818 : listconstruct
Adding op '818' of type const
Converting op 819 : listconstruct
Adding op '819' of type const
Converting op 820 : listconstruct
Adding op '820' of type const
Converting op x.85 : _convolution
Adding op 'x.85' of type conv
Adding op 'x.85_pad_type_0' of type const
Adding op 'x.85_pad_0' of type const
Converting op 822 : sigmoid
Adding op '822' of type sigmoid
Converting op input.75 : mul
Adding op 'input.75' of type mul
Converting op 827 : listconstruct
Adding op '827' of type const
Converting op 828 : listconstruct
Adding op '828' of type const
Converting op 829 : listconstruct
Adding op '829' of type const
Converting op 830 : listconstruct
Adding op '830' of type const
Converting op x.87 : _convolution
Adding op 'x.87' of type conv
Adding op 'x.87_pad_type_0' of type const
Adding op 'x.87_pad_0' of type const
Converting op 832 : sigmoid
Adding op '832' of type sigmoid
Converting op input.77 : mul
Adding op 'input.77' of type mul
Converting op 837 : listconstruct
Adding op '837' of type const
Converting op 838 : listconstruct
Adding op '838' of type const
Converting op 839 : listconstruct
Adding op '839' of type const
Converting op 840 : listconstruct
Adding op '840' of type const
Converting op x.89 : _convolution
Adding op 'x.89' of type conv
Adding op 'x.89_pad_type_0' of type const
Adding op 'x.89_pad_0' of type const
Converting op 842 : sigmoid
Adding op '842' of type sigmoid
Converting op input.79 : mul
Adding op 'input.79' of type mul
Converting op 844 : listconstruct
Adding op '844' of type const
Converting op 845 : listconstruct
Adding op '845' of type const
Converting op 846 : listconstruct
Adding op '846' of type const
Converting op 847 : listconstruct
Adding op '847' of type const
Converting op 848 : max_pool2d
Adding op '848' of type max_pool
Adding op '848_pad_type_0' of type const
Adding op '848_pad_0' of type const
Adding op '848_ceil_mode_0' of type const
Converting op 849 : listconstruct
Adding op '849' of type const
Converting op 850 : listconstruct
Adding op '850' of type const
Converting op 851 : listconstruct
Adding op '851' of type const
Converting op 852 : listconstruct
Adding op '852' of type const
Converting op 853 : max_pool2d
Adding op '853' of type max_pool
Adding op '853_pad_type_0' of type const
Adding op '853_pad_0' of type const
Adding op '853_ceil_mode_0' of type const
Converting op 854 : listconstruct
Adding op '854' of type const
Converting op 855 : listconstruct
Adding op '855' of type const
Converting op 856 : listconstruct
Adding op '856' of type const
Converting op 857 : listconstruct
Adding op '857' of type const
Converting op 858 : max_pool2d
Adding op '858' of type max_pool
Adding op '858_pad_type_0' of type const
Adding op '858_pad_0' of type const
Adding op '858_ceil_mode_0' of type const
Converting op 859 : listconstruct
Converting op input.81 : cat
Adding op 'input.81' of type concat
Adding op 'input.81_interleave_0' of type const
Converting op 864 : listconstruct
Adding op '864' of type const
Converting op 865 : listconstruct
Adding op '865' of type const
Converting op 866 : listconstruct
Adding op '866' of type const
Converting op 867 : listconstruct
Adding op '867' of type const
Converting op x.91 : _convolution
Adding op 'x.91' of type conv
Adding op 'x.91_pad_type_0' of type const
Adding op 'x.91_pad_0' of type const
Converting op 869 : sigmoid
Adding op '869' of type sigmoid
Converting op input.83 : mul
Adding op 'input.83' of type mul
Converting op 874 : listconstruct
Adding op '874' of type const
Converting op 875 : listconstruct
Adding op '875' of type const
Converting op 876 : listconstruct
Adding op '876' of type const
Converting op 877 : listconstruct
Adding op '877' of type const
Converting op x.93 : _convolution
Adding op 'x.93' of type conv
Adding op 'x.93_pad_type_0' of type const
Adding op 'x.93_pad_0' of type const
Converting op 879 : sigmoid
Adding op '879' of type sigmoid
Converting op y1 : mul
Adding op 'y1' of type mul
Converting op 884 : listconstruct
Adding op '884' of type const
Converting op 885 : listconstruct
Adding op '885' of type const
Converting op 886 : listconstruct
Adding op '886' of type const
Converting op 887 : listconstruct
Adding op '887' of type const
Converting op x.95 : _convolution
Adding op 'x.95' of type conv
Adding op 'x.95_pad_type_0' of type const
Adding op 'x.95_pad_0' of type const
Converting op 889 : sigmoid
Adding op '889' of type sigmoid
Converting op y2 : mul
Adding op 'y2' of type mul
Converting op 891 : listconstruct
Converting op input.85 : cat
Adding op 'input.85' of type concat
Adding op 'input.85_interleave_0' of type const
Converting op 896 : listconstruct
Adding op '896' of type const
Converting op 897 : listconstruct
Adding op '897' of type const
Converting op 898 : listconstruct
Adding op '898' of type const
Converting op 899 : listconstruct
Adding op '899' of type const
Converting op x.97 : _convolution
Adding op 'x.97' of type conv
Adding op 'x.97_pad_type_0' of type const
Adding op 'x.97_pad_0' of type const
Converting op 901 : sigmoid
Adding op '901' of type sigmoid
Converting op input.87 : mul
Adding op 'input.87' of type mul
Converting op 903 : constant
Adding op '903' of type const
Converting op 904 : constant
Adding op '904' of type const
Converting op 905 : constant
Adding op '905' of type const
Converting op 906 : constant
Adding op '906' of type const
Converting op 910 : listconstruct
Adding op '910' of type const
Converting op 911 : listconstruct
Adding op '911' of type const
Converting op 912 : listconstruct
Adding op '912' of type const
Converting op 913 : listconstruct
Adding op '913' of type const
Converting op x.99 : _convolution
Adding op 'x.99' of type conv
Adding op 'x.99_pad_type_0' of type const
Adding op 'x.99_pad_0' of type const
Converting op 915 : sigmoid
Adding op '915' of type sigmoid
Converting op input.89 : mul
Adding op 'input.89' of type mul
Converting op 917 : constant
Adding op '917' of type const
Converting op 918 : constant
Converting op 919 : listconstruct
Adding op '919' of type const
Converting op 920 : upsample_nearest2d
Adding op '920' of type upsample_nearest_neighbor
Adding op '920_scale_factor_height_0' of type const
Adding op '920_scale_factor_width_0' of type const
Converting op 921 : constant
Adding op '921' of type const
Converting op 922 : constant
Adding op '922' of type const
Converting op 923 : constant
Adding op '923' of type const
Converting op 924 : constant
Adding op '924' of type const
Converting op 928 : listconstruct
Adding op '928' of type const
Converting op 929 : listconstruct
Adding op '929' of type const
Converting op 930 : listconstruct
Adding op '930' of type const
Converting op 931 : listconstruct
Adding op '931' of type const
Converting op x.101 : _convolution
Adding op 'x.101' of type conv
Adding op 'x.101_pad_type_0' of type const
Adding op 'x.101_pad_0' of type const
Converting op 933 : sigmoid
Adding op '933' of type sigmoid
Converting op 934 : mul
Adding op '934' of type mul
Converting op 935 : constant
Adding op '935' of type const
Converting op 936 : listconstruct
Converting op input.91 : cat
Adding op 'input.91' of type concat
Adding op 'input.91_interleave_0' of type const
Converting op 938 : constant
Adding op '938' of type const
Converting op 939 : constant
Adding op '939' of type const
Converting op 940 : constant
Adding op '940' of type const
Converting op 941 : constant
Adding op '941' of type const
Converting op 945 : listconstruct
Adding op '945' of type const
Converting op 946 : listconstruct
Adding op '946' of type const
Converting op 947 : listconstruct
Adding op '947' of type const
Converting op 948 : listconstruct
Adding op '948' of type const
Converting op x.103 : _convolution
Adding op 'x.103' of type conv
Adding op 'x.103_pad_type_0' of type const
Adding op 'x.103_pad_0' of type const
Converting op 950 : sigmoid
Adding op '950' of type sigmoid
Converting op 951 : mul
Adding op '951' of type mul
Converting op 952 : constant
Adding op '952' of type const
Converting op 953 : constant
Adding op '953' of type const
Converting op 954 : constant
Adding op '954' of type const
Converting op 955 : constant
Adding op '955' of type const
Converting op 959 : listconstruct
Adding op '959' of type const
Converting op 960 : listconstruct
Adding op '960' of type const
Converting op 961 : listconstruct
Adding op '961' of type const
Converting op 962 : listconstruct
Adding op '962' of type const
Converting op x.105 : _convolution
Adding op 'x.105' of type conv
Adding op 'x.105_pad_type_0' of type const
Adding op 'x.105_pad_0' of type const
Converting op 964 : sigmoid
Adding op '964' of type sigmoid
Converting op input.93 : mul
Adding op 'input.93' of type mul
Converting op 966 : constant
Adding op '966' of type const
Converting op 967 : constant
Adding op '967' of type const
Converting op 968 : constant
Adding op '968' of type const
Converting op 969 : constant
Adding op '969' of type const
Converting op 973 : listconstruct
Adding op '973' of type const
Converting op 974 : listconstruct
Adding op '974' of type const
Converting op 975 : listconstruct
Adding op '975' of type const
Converting op 976 : listconstruct
Adding op '976' of type const
Converting op x.107 : _convolution
Adding op 'x.107' of type conv
Adding op 'x.107_pad_type_0' of type const
Adding op 'x.107_pad_0' of type const
Converting op 978 : sigmoid
Adding op '978' of type sigmoid
Converting op input.95 : mul
Adding op 'input.95' of type mul
Converting op 980 : constant
Adding op '980' of type const
Converting op 981 : constant
Adding op '981' of type const
Converting op 982 : constant
Adding op '982' of type const
Converting op 983 : constant
Adding op '983' of type const
Converting op 987 : listconstruct
Adding op '987' of type const
Converting op 988 : listconstruct
Adding op '988' of type const
Converting op 989 : listconstruct
Adding op '989' of type const
Converting op 990 : listconstruct
Adding op '990' of type const
Converting op x.109 : _convolution
Adding op 'x.109' of type conv
Adding op 'x.109_pad_type_0' of type const
Adding op 'x.109_pad_0' of type const
Converting op 992 : sigmoid
Adding op '992' of type sigmoid
Converting op input.97 : mul
Adding op 'input.97' of type mul
Converting op 994 : constant
Adding op '994' of type const
Converting op 995 : constant
Adding op '995' of type const
Converting op 996 : constant
Adding op '996' of type const
Converting op 997 : constant
Adding op '997' of type const
Converting op 1001 : listconstruct
Adding op '1001' of type const
Converting op 1002 : listconstruct
Adding op '1002' of type const
Converting op 1003 : listconstruct
Adding op '1003' of type const
Converting op 1004 : listconstruct
Adding op '1004' of type const
Converting op x.111 : _convolution
Adding op 'x.111' of type conv
Adding op 'x.111_pad_type_0' of type const
Adding op 'x.111_pad_0' of type const
Converting op 1006 : sigmoid
Adding op '1006' of type sigmoid
Converting op input.99 : mul
Adding op 'input.99' of type mul
Converting op 1008 : constant
Adding op '1008' of type const
Converting op 1009 : constant
Adding op '1009' of type const
Converting op 1010 : constant
Adding op '1010' of type const
Converting op 1011 : constant
Adding op '1011' of type const
Converting op 1015 : listconstruct
Adding op '1015' of type const
Converting op 1016 : listconstruct
Adding op '1016' of type const
Converting op 1017 : listconstruct
Adding op '1017' of type const
Converting op 1018 : listconstruct
Adding op '1018' of type const
Converting op x.113 : _convolution
Adding op 'x.113' of type conv
Adding op 'x.113_pad_type_0' of type const
Adding op 'x.113_pad_0' of type const
Converting op 1020 : sigmoid
Adding op '1020' of type sigmoid
Converting op 1021 : mul
Adding op '1021' of type mul
Converting op 1022 : constant
Adding op '1022' of type const
Converting op 1023 : listconstruct
Converting op input.101 : cat
Adding op 'input.101' of type concat
Adding op 'input.101_interleave_0' of type const
Converting op 1025 : constant
Adding op '1025' of type const
Converting op 1026 : constant
Adding op '1026' of type const
Converting op 1027 : constant
Adding op '1027' of type const
Converting op 1028 : constant
Adding op '1028' of type const
Converting op 1032 : listconstruct
Adding op '1032' of type const
Converting op 1033 : listconstruct
Adding op '1033' of type const
Converting op 1034 : listconstruct
Adding op '1034' of type const
Converting op 1035 : listconstruct
Adding op '1035' of type const
Converting op x.115 : _convolution
Adding op 'x.115' of type conv
Adding op 'x.115_pad_type_0' of type const
Adding op 'x.115_pad_0' of type const
Converting op 1037 : sigmoid
Adding op '1037' of type sigmoid
Converting op input.103 : mul
Adding op 'input.103' of type mul
Converting op 1039 : constant
Adding op '1039' of type const
Converting op 1040 : constant
Adding op '1040' of type const
Converting op 1041 : constant
Adding op '1041' of type const
Converting op 1042 : constant
Adding op '1042' of type const
Converting PyTorch Frontend ==> MIL Ops:  58%|███████████████████▋              | 695/1203 [00:00<00:00, 2203.58 ops/s]Converting op 1046 : listconstruct
Adding op '1046' of type const
Converting op 1047 : listconstruct
Adding op '1047' of type const
Converting op 1048 : listconstruct
Adding op '1048' of type const
Converting op 1049 : listconstruct
Adding op '1049' of type const
Converting op x.117 : _convolution
Adding op 'x.117' of type conv
Adding op 'x.117_pad_type_0' of type const
Adding op 'x.117_pad_0' of type const
Converting op 1051 : sigmoid
Adding op '1051' of type sigmoid
Converting op input.105 : mul
Adding op 'input.105' of type mul
Converting op 1053 : constant
Adding op '1053' of type const
Converting op 1054 : constant
Converting op 1055 : listconstruct
Adding op '1055' of type const
Converting op 1056 : upsample_nearest2d
Adding op '1056' of type upsample_nearest_neighbor
Adding op '1056_scale_factor_height_0' of type const
Adding op '1056_scale_factor_width_0' of type const
Converting op 1057 : constant
Adding op '1057' of type const
Converting op 1058 : constant
Adding op '1058' of type const
Converting op 1059 : constant
Adding op '1059' of type const
Converting op 1060 : constant
Adding op '1060' of type const
Converting op 1064 : listconstruct
Adding op '1064' of type const
Converting op 1065 : listconstruct
Adding op '1065' of type const
Converting op 1066 : listconstruct
Adding op '1066' of type const
Converting op 1067 : listconstruct
Adding op '1067' of type const
Converting op x.119 : _convolution
Adding op 'x.119' of type conv
Adding op 'x.119_pad_type_0' of type const
Adding op 'x.119_pad_0' of type const
Converting op 1069 : sigmoid
Adding op '1069' of type sigmoid
Converting op 1070 : mul
Adding op '1070' of type mul
Converting op 1071 : constant
Adding op '1071' of type const
Converting op 1072 : listconstruct
Converting op input.107 : cat
Adding op 'input.107' of type concat
Adding op 'input.107_interleave_0' of type const
Converting op 1074 : constant
Adding op '1074' of type const
Converting op 1075 : constant
Adding op '1075' of type const
Converting op 1076 : constant
Adding op '1076' of type const
Converting op 1077 : constant
Adding op '1077' of type const
Converting op 1081 : listconstruct
Adding op '1081' of type const
Converting op 1082 : listconstruct
Adding op '1082' of type const
Converting op 1083 : listconstruct
Adding op '1083' of type const
Converting op 1084 : listconstruct
Adding op '1084' of type const
Converting op x.121 : _convolution
Adding op 'x.121' of type conv
Adding op 'x.121_pad_type_0' of type const
Adding op 'x.121_pad_0' of type const
Converting op 1086 : sigmoid
Adding op '1086' of type sigmoid
Converting op 1087 : mul
Adding op '1087' of type mul
Converting op 1088 : constant
Adding op '1088' of type const
Converting op 1089 : constant
Adding op '1089' of type const
Converting op 1090 : constant
Adding op '1090' of type const
Converting op 1091 : constant
Adding op '1091' of type const
Converting op 1095 : listconstruct
Adding op '1095' of type const
Converting op 1096 : listconstruct
Adding op '1096' of type const
Converting op 1097 : listconstruct
Adding op '1097' of type const
Converting op 1098 : listconstruct
Adding op '1098' of type const
Converting op x.123 : _convolution
Adding op 'x.123' of type conv
Adding op 'x.123_pad_type_0' of type const
Adding op 'x.123_pad_0' of type const
Converting op 1100 : sigmoid
Adding op '1100' of type sigmoid
Converting op input.109 : mul
Adding op 'input.109' of type mul
Converting op 1102 : constant
Adding op '1102' of type const
Converting op 1103 : constant
Adding op '1103' of type const
Converting op 1104 : constant
Adding op '1104' of type const
Converting op 1105 : constant
Adding op '1105' of type const
Converting op 1109 : listconstruct
Adding op '1109' of type const
Converting op 1110 : listconstruct
Adding op '1110' of type const
Converting op 1111 : listconstruct
Adding op '1111' of type const
Converting op 1112 : listconstruct
Adding op '1112' of type const
Converting op x.125 : _convolution
Adding op 'x.125' of type conv
Adding op 'x.125_pad_type_0' of type const
Adding op 'x.125_pad_0' of type const
Converting op 1114 : sigmoid
Adding op '1114' of type sigmoid
Converting op input.111 : mul
Adding op 'input.111' of type mul
Converting op 1116 : constant
Adding op '1116' of type const
Converting op 1117 : constant
Adding op '1117' of type const
Converting op 1118 : constant
Adding op '1118' of type const
Converting op 1119 : constant
Adding op '1119' of type const
Converting op 1123 : listconstruct
Adding op '1123' of type const
Converting op 1124 : listconstruct
Adding op '1124' of type const
Converting op 1125 : listconstruct
Adding op '1125' of type const
Converting op 1126 : listconstruct
Adding op '1126' of type const
Converting op x.127 : _convolution
Adding op 'x.127' of type conv
Adding op 'x.127_pad_type_0' of type const
Adding op 'x.127_pad_0' of type const
Converting op 1128 : sigmoid
Adding op '1128' of type sigmoid
Converting op input.113 : mul
Adding op 'input.113' of type mul
Converting op 1130 : constant
Adding op '1130' of type const
Converting op 1131 : constant
Adding op '1131' of type const
Converting op 1132 : constant
Adding op '1132' of type const
Converting op 1133 : constant
Adding op '1133' of type const
Converting op 1137 : listconstruct
Adding op '1137' of type const
Converting op 1138 : listconstruct
Adding op '1138' of type const
Converting op 1139 : listconstruct
Adding op '1139' of type const
Converting op 1140 : listconstruct
Adding op '1140' of type const
Converting op x.129 : _convolution
Adding op 'x.129' of type conv
Adding op 'x.129_pad_type_0' of type const
Adding op 'x.129_pad_0' of type const
Converting op 1142 : sigmoid
Adding op '1142' of type sigmoid
Converting op input.115 : mul
Adding op 'input.115' of type mul
Converting op 1144 : constant
Adding op '1144' of type const
Converting op 1145 : constant
Adding op '1145' of type const
Converting op 1146 : constant
Adding op '1146' of type const
Converting op 1147 : constant
Adding op '1147' of type const
Converting op 1151 : listconstruct
Adding op '1151' of type const
Converting op 1152 : listconstruct
Adding op '1152' of type const
Converting op 1153 : listconstruct
Adding op '1153' of type const
Converting op 1154 : listconstruct
Adding op '1154' of type const
Converting op x.131 : _convolution
Adding op 'x.131' of type conv
Adding op 'x.131_pad_type_0' of type const
Adding op 'x.131_pad_0' of type const
Converting op 1156 : sigmoid
Adding op '1156' of type sigmoid
Converting op 1157 : mul
Adding op '1157' of type mul
Converting op 1158 : constant
Adding op '1158' of type const
Converting op 1159 : listconstruct
Converting op input.117 : cat
Adding op 'input.117' of type concat
Adding op 'input.117_interleave_0' of type const
Converting op 1161 : constant
Adding op '1161' of type const
Converting op 1162 : constant
Adding op '1162' of type const
Converting op 1163 : constant
Adding op '1163' of type const
Converting op 1164 : constant
Adding op '1164' of type const
Converting op 1168 : listconstruct
Adding op '1168' of type const
Converting op 1169 : listconstruct
Adding op '1169' of type const
Converting op 1170 : listconstruct
Adding op '1170' of type const
Converting op 1171 : listconstruct
Adding op '1171' of type const
Converting op x.133 : _convolution
Adding op 'x.133' of type conv
Adding op 'x.133_pad_type_0' of type const
Adding op 'x.133_pad_0' of type const
Converting op 1173 : sigmoid
Adding op '1173' of type sigmoid
Converting op input.119 : mul
Adding op 'input.119' of type mul
Converting op 1175 : constant
Adding op '1175' of type const
Converting op 1176 : constant
Adding op '1176' of type const
Converting op 1177 : constant
Adding op '1177' of type const
Converting op 1178 : constant
Adding op '1178' of type const
Converting op 1179 : listconstruct
Adding op '1179' of type const
Converting op 1180 : listconstruct
Adding op '1180' of type const
Converting op 1181 : listconstruct
Adding op '1181' of type const
Converting op 1182 : listconstruct
Adding op '1182' of type const
Converting op input.121 : max_pool2d
Adding op 'input.121' of type max_pool
Adding op 'input.121_pad_type_0' of type const
Adding op 'input.121_pad_0' of type const
Adding op 'input.121_ceil_mode_0' of type const
Converting op 1184 : constant
Adding op '1184' of type const
Converting op 1185 : constant
Adding op '1185' of type const
Converting op 1186 : constant
Adding op '1186' of type const
Converting op 1187 : constant
Adding op '1187' of type const
Converting op 1191 : listconstruct
Adding op '1191' of type const
Converting op 1192 : listconstruct
Adding op '1192' of type const
Converting op 1193 : listconstruct
Adding op '1193' of type const
Converting op 1194 : listconstruct
Adding op '1194' of type const
Converting op x.135 : _convolution
Adding op 'x.135' of type conv
Adding op 'x.135_pad_type_0' of type const
Adding op 'x.135_pad_0' of type const
Converting op 1196 : sigmoid
Adding op '1196' of type sigmoid
Converting op 1197 : mul
Adding op '1197' of type mul
Converting op 1198 : constant
Adding op '1198' of type const
Converting op 1199 : constant
Adding op '1199' of type const
Converting op 1200 : constant
Adding op '1200' of type const
Converting op 1201 : constant
Adding op '1201' of type const
Converting op 1205 : listconstruct
Adding op '1205' of type const
Converting op 1206 : listconstruct
Adding op '1206' of type const
Converting op 1207 : listconstruct
Adding op '1207' of type const
Converting op 1208 : listconstruct
Adding op '1208' of type const
Converting op x.137 : _convolution
Adding op 'x.137' of type conv
Adding op 'x.137_pad_type_0' of type const
Adding op 'x.137_pad_0' of type const
Converting op 1210 : sigmoid
Adding op '1210' of type sigmoid
Converting op input.123 : mul
Adding op 'input.123' of type mul
Converting op 1212 : constant
Adding op '1212' of type const
Converting op 1213 : constant
Adding op '1213' of type const
Converting op 1214 : constant
Adding op '1214' of type const
Converting op 1215 : constant
Adding op '1215' of type const
Converting op 1216 : constant
Adding op '1216' of type const
Converting op 1220 : listconstruct
Adding op '1220' of type const
Converting op 1221 : listconstruct
Adding op '1221' of type const
Converting op 1222 : listconstruct
Adding op '1222' of type const
Converting op 1223 : listconstruct
Adding op '1223' of type const
Converting op x.139 : _convolution
Adding op 'x.139' of type conv
Adding op 'x.139_pad_type_0' of type const
Adding op 'x.139_pad_0' of type const
Converting op 1225 : sigmoid
Adding op '1225' of type sigmoid
Converting op 1226 : mul
Adding op '1226' of type mul
Converting op 1227 : constant
Adding op '1227' of type const
Converting op 1228 : listconstruct
Converting op input.125 : cat
Adding op 'input.125' of type concat
Adding op 'input.125_interleave_0' of type const
Converting op 1230 : constant
Adding op '1230' of type const
Converting op 1231 : constant
Adding op '1231' of type const
Converting op 1232 : constant
Adding op '1232' of type const
Converting op 1233 : constant
Adding op '1233' of type const
Converting op 1237 : listconstruct
Adding op '1237' of type const
Converting op 1238 : listconstruct
Adding op '1238' of type const
Converting op 1239 : listconstruct
Adding op '1239' of type const
Converting op 1240 : listconstruct
Adding op '1240' of type const
Converting op x.141 : _convolution
Adding op 'x.141' of type conv
Adding op 'x.141_pad_type_0' of type const
Adding op 'x.141_pad_0' of type const
Converting op 1242 : sigmoid
Adding op '1242' of type sigmoid
Converting op 1243 : mul
Adding op '1243' of type mul
Converting op 1244 : constant
Adding op '1244' of type const
Converting op 1245 : constant
Adding op '1245' of type const
Converting op 1246 : constant
Adding op '1246' of type const
Converting op 1247 : constant
Adding op '1247' of type const
Converting op 1251 : listconstruct
Adding op '1251' of type const
Converting op 1252 : listconstruct
Adding op '1252' of type const
Converting op 1253 : listconstruct
Adding op '1253' of type const
Converting op 1254 : listconstruct
Adding op '1254' of type const
Converting op x.143 : _convolution
Adding op 'x.143' of type conv
Adding op 'x.143_pad_type_0' of type const
Adding op 'x.143_pad_0' of type const
Converting op 1256 : sigmoid
Adding op '1256' of type sigmoid
Converting op input.127 : mul
Adding op 'input.127' of type mul
Converting op 1258 : constant
Adding op '1258' of type const
Converting op 1259 : constant
Adding op '1259' of type const
Converting op 1260 : constant
Adding op '1260' of type const
Converting op 1261 : constant
Adding op '1261' of type const
Converting op 1265 : listconstruct
Adding op '1265' of type const
Converting op 1266 : listconstruct
Adding op '1266' of type const
Converting op 1267 : listconstruct
Adding op '1267' of type const
Converting op 1268 : listconstruct
Adding op '1268' of type const
Converting op x.145 : _convolution
Adding op 'x.145' of type conv
Adding op 'x.145_pad_type_0' of type const
Adding op 'x.145_pad_0' of type const
Converting op 1270 : sigmoid
Adding op '1270' of type sigmoid
Converting op input.129 : mul
Adding op 'input.129' of type mul
Converting op 1272 : constant
Adding op '1272' of type const
Converting op 1273 : constant
Adding op '1273' of type const
Converting op 1274 : constant
Adding op '1274' of type const
Converting op 1275 : constant
Adding op '1275' of type const
Converting op 1279 : listconstruct
Adding op '1279' of type const
Converting op 1280 : listconstruct
Adding op '1280' of type const
Converting op 1281 : listconstruct
Adding op '1281' of type const
Converting op 1282 : listconstruct
Adding op '1282' of type const
Converting op x.147 : _convolution
Adding op 'x.147' of type conv
Adding op 'x.147_pad_type_0' of type const
Adding op 'x.147_pad_0' of type const
Converting op 1284 : sigmoid
Adding op '1284' of type sigmoid
Converting op input.131 : mul
Adding op 'input.131' of type mul
Converting op 1286 : constant
Adding op '1286' of type const
Converting op 1287 : constant
Adding op '1287' of type const
Converting op 1288 : constant
Adding op '1288' of type const
Converting op 1289 : constant
Adding op '1289' of type const
Converting op 1293 : listconstruct
Adding op '1293' of type const
Converting op 1294 : listconstruct
Adding op '1294' of type const
Converting op 1295 : listconstruct
Adding op '1295' of type const
Converting op 1296 : listconstruct
Adding op '1296' of type const
Converting op x.149 : _convolution
Adding op 'x.149' of type conv
Adding op 'x.149_pad_type_0' of type const
Adding op 'x.149_pad_0' of type const
Converting op 1298 : sigmoid
Adding op '1298' of type sigmoid
Converting op input.133 : mul
Adding op 'input.133' of type mul
Converting op 1300 : constant
Adding op '1300' of type const
Converting op 1301 : constant
Adding op '1301' of type const
Converting op 1302 : constant
Adding op '1302' of type const
Converting op 1303 : constant
Adding op '1303' of type const
Converting op 1307 : listconstruct
Adding op '1307' of type const
Converting op 1308 : listconstruct
Adding op '1308' of type const
Converting op 1309 : listconstruct
Adding op '1309' of type const
Converting op 1310 : listconstruct
Adding op '1310' of type const
Converting op x.151 : _convolution
Adding op 'x.151' of type conv
Adding op 'x.151_pad_type_0' of type const
Adding op 'x.151_pad_0' of type const
Converting op 1312 : sigmoid
Adding op '1312' of type sigmoid
Converting op 1313 : mul
Adding op '1313' of type mul
Converting op 1314 : constant
Adding op '1314' of type const
Converting op 1315 : listconstruct
Converting op input.135 : cat
Adding op 'input.135' of type concat
Adding op 'input.135_interleave_0' of type const
Converting op 1317 : constant
Adding op '1317' of type const
Converting op 1318 : constant
Adding op '1318' of type const
Converting op 1319 : constant
Adding op '1319' of type const
Converting op 1320 : constant
Adding op '1320' of type const
Converting op 1324 : listconstruct
Adding op '1324' of type const
Converting op 1325 : listconstruct
Adding op '1325' of type const
Converting op 1326 : listconstruct
Adding op '1326' of type const
Converting op 1327 : listconstruct
Adding op '1327' of type const
Converting op x.153 : _convolution
Adding op 'x.153' of type conv
Adding op 'x.153_pad_type_0' of type const
Adding op 'x.153_pad_0' of type const
Converting PyTorch Frontend ==> MIL Ops:  77%|██████████████████████████        | 924/1203 [00:00<00:00, 2231.81 ops/s]Converting op 1329 : sigmoid
Adding op '1329' of type sigmoid
Converting op input.137 : mul
Adding op 'input.137' of type mul
Converting op 1331 : constant
Adding op '1331' of type const
Converting op 1332 : constant
Adding op '1332' of type const
Converting op 1333 : constant
Adding op '1333' of type const
Converting op 1334 : constant
Adding op '1334' of type const
Converting op 1335 : listconstruct
Adding op '1335' of type const
Converting op 1336 : listconstruct
Adding op '1336' of type const
Converting op 1337 : listconstruct
Adding op '1337' of type const
Converting op 1338 : listconstruct
Adding op '1338' of type const
Converting op input.139 : max_pool2d
Adding op 'input.139' of type max_pool
Adding op 'input.139_pad_type_0' of type const
Adding op 'input.139_pad_0' of type const
Adding op 'input.139_ceil_mode_0' of type const
Converting op 1340 : constant
Adding op '1340' of type const
Converting op 1341 : constant
Adding op '1341' of type const
Converting op 1342 : constant
Adding op '1342' of type const
Converting op 1343 : constant
Adding op '1343' of type const
Converting op 1347 : listconstruct
Adding op '1347' of type const
Converting op 1348 : listconstruct
Adding op '1348' of type const
Converting op 1349 : listconstruct
Adding op '1349' of type const
Converting op 1350 : listconstruct
Adding op '1350' of type const
Converting op x.155 : _convolution
Adding op 'x.155' of type conv
Adding op 'x.155_pad_type_0' of type const
Adding op 'x.155_pad_0' of type const
Converting op 1352 : sigmoid
Adding op '1352' of type sigmoid
Converting op 1353 : mul
Adding op '1353' of type mul
Converting op 1354 : constant
Adding op '1354' of type const
Converting op 1355 : constant
Adding op '1355' of type const
Converting op 1356 : constant
Adding op '1356' of type const
Converting op 1357 : constant
Adding op '1357' of type const
Converting op 1361 : listconstruct
Adding op '1361' of type const
Converting op 1362 : listconstruct
Adding op '1362' of type const
Converting op 1363 : listconstruct
Adding op '1363' of type const
Converting op 1364 : listconstruct
Adding op '1364' of type const
Converting op x.157 : _convolution
Adding op 'x.157' of type conv
Adding op 'x.157_pad_type_0' of type const
Adding op 'x.157_pad_0' of type const
Converting op 1366 : sigmoid
Adding op '1366' of type sigmoid
Converting op input.141 : mul
Adding op 'input.141' of type mul
Converting op 1368 : constant
Adding op '1368' of type const
Converting op 1369 : constant
Adding op '1369' of type const
Converting op 1370 : constant
Adding op '1370' of type const
Converting op 1371 : constant
Adding op '1371' of type const
Converting op 1372 : constant
Adding op '1372' of type const
Converting op 1376 : listconstruct
Adding op '1376' of type const
Converting op 1377 : listconstruct
Adding op '1377' of type const
Converting op 1378 : listconstruct
Adding op '1378' of type const
Converting op 1379 : listconstruct
Adding op '1379' of type const
Converting op x.159 : _convolution
Adding op 'x.159' of type conv
Adding op 'x.159_pad_type_0' of type const
Adding op 'x.159_pad_0' of type const
Converting op 1381 : sigmoid
Adding op '1381' of type sigmoid
Converting op 1382 : mul
Adding op '1382' of type mul
Converting op 1383 : constant
Adding op '1383' of type const
Converting op 1384 : listconstruct
Converting op input.143 : cat
Adding op 'input.143' of type concat
Adding op 'input.143_interleave_0' of type const
Converting op 1386 : constant
Adding op '1386' of type const
Converting op 1387 : constant
Adding op '1387' of type const
Converting op 1388 : constant
Adding op '1388' of type const
Converting op 1389 : constant
Adding op '1389' of type const
Converting op 1393 : listconstruct
Adding op '1393' of type const
Converting op 1394 : listconstruct
Adding op '1394' of type const
Converting op 1395 : listconstruct
Adding op '1395' of type const
Converting op 1396 : listconstruct
Adding op '1396' of type const
Converting op x.161 : _convolution
Adding op 'x.161' of type conv
Adding op 'x.161_pad_type_0' of type const
Adding op 'x.161_pad_0' of type const
Converting op 1398 : sigmoid
Adding op '1398' of type sigmoid
Converting op 1399 : mul
Adding op '1399' of type mul
Converting op 1400 : constant
Adding op '1400' of type const
Converting op 1401 : constant
Adding op '1401' of type const
Converting op 1402 : constant
Adding op '1402' of type const
Converting op 1403 : constant
Adding op '1403' of type const
Converting op 1407 : listconstruct
Adding op '1407' of type const
Converting op 1408 : listconstruct
Adding op '1408' of type const
Converting op 1409 : listconstruct
Adding op '1409' of type const
Converting op 1410 : listconstruct
Adding op '1410' of type const
Converting op x.163 : _convolution
Adding op 'x.163' of type conv
Adding op 'x.163_pad_type_0' of type const
Adding op 'x.163_pad_0' of type const
Converting op 1412 : sigmoid
Adding op '1412' of type sigmoid
Converting op input.145 : mul
Adding op 'input.145' of type mul
Converting op 1414 : constant
Adding op '1414' of type const
Converting op 1415 : constant
Adding op '1415' of type const
Converting op 1416 : constant
Adding op '1416' of type const
Converting op 1417 : constant
Adding op '1417' of type const
Converting op 1421 : listconstruct
Adding op '1421' of type const
Converting op 1422 : listconstruct
Adding op '1422' of type const
Converting op 1423 : listconstruct
Adding op '1423' of type const
Converting op 1424 : listconstruct
Adding op '1424' of type const
Converting op x.165 : _convolution
Adding op 'x.165' of type conv
Adding op 'x.165_pad_type_0' of type const
Adding op 'x.165_pad_0' of type const
Converting op 1426 : sigmoid
Adding op '1426' of type sigmoid
Converting op input.147 : mul
Adding op 'input.147' of type mul
Converting op 1428 : constant
Adding op '1428' of type const
Converting op 1429 : constant
Adding op '1429' of type const
Converting op 1430 : constant
Adding op '1430' of type const
Converting op 1431 : constant
Adding op '1431' of type const
Converting op 1435 : listconstruct
Adding op '1435' of type const
Converting op 1436 : listconstruct
Adding op '1436' of type const
Converting op 1437 : listconstruct
Adding op '1437' of type const
Converting op 1438 : listconstruct
Adding op '1438' of type const
Converting op x.167 : _convolution
Adding op 'x.167' of type conv
Adding op 'x.167_pad_type_0' of type const
Adding op 'x.167_pad_0' of type const
Converting op 1440 : sigmoid
Adding op '1440' of type sigmoid
Converting op input.149 : mul
Adding op 'input.149' of type mul
Converting op 1442 : constant
Adding op '1442' of type const
Converting op 1443 : constant
Adding op '1443' of type const
Converting op 1444 : constant
Adding op '1444' of type const
Converting op 1445 : constant
Adding op '1445' of type const
Converting op 1449 : listconstruct
Adding op '1449' of type const
Converting op 1450 : listconstruct
Adding op '1450' of type const
Converting op 1451 : listconstruct
Adding op '1451' of type const
Converting op 1452 : listconstruct
Adding op '1452' of type const
Converting op x.169 : _convolution
Adding op 'x.169' of type conv
Adding op 'x.169_pad_type_0' of type const
Adding op 'x.169_pad_0' of type const
Converting op 1454 : sigmoid
Adding op '1454' of type sigmoid
Converting op input.151 : mul
Adding op 'input.151' of type mul
Converting op 1456 : constant
Adding op '1456' of type const
Converting op 1457 : constant
Adding op '1457' of type const
Converting op 1458 : constant
Adding op '1458' of type const
Converting op 1459 : constant
Adding op '1459' of type const
Converting op 1463 : listconstruct
Adding op '1463' of type const
Converting op 1464 : listconstruct
Adding op '1464' of type const
Converting op 1465 : listconstruct
Adding op '1465' of type const
Converting op 1466 : listconstruct
Adding op '1466' of type const
Converting op x.171 : _convolution
Adding op 'x.171' of type conv
Adding op 'x.171_pad_type_0' of type const
Adding op 'x.171_pad_0' of type const
Converting op 1468 : sigmoid
Adding op '1468' of type sigmoid
Converting op 1469 : mul
Adding op '1469' of type mul
Converting op 1470 : constant
Adding op '1470' of type const
Converting op 1471 : listconstruct
Converting op input.153 : cat
Adding op 'input.153' of type concat
Adding op 'input.153_interleave_0' of type const
Converting op 1473 : constant
Adding op '1473' of type const
Converting op 1474 : constant
Adding op '1474' of type const
Converting op 1475 : constant
Adding op '1475' of type const
Converting op 1476 : constant
Adding op '1476' of type const
Converting op 1480 : listconstruct
Adding op '1480' of type const
Converting op 1481 : listconstruct
Adding op '1481' of type const
Converting op 1482 : listconstruct
Adding op '1482' of type const
Converting op 1483 : listconstruct
Adding op '1483' of type const
Converting op x : _convolution
Adding op 'x' of type conv
Adding op 'x_pad_type_0' of type const
Adding op 'x_pad_0' of type const
Converting op 1485 : sigmoid
Adding op '1485' of type sigmoid
Converting op input.159 : mul
Adding op 'input.159' of type mul
Converting op 1487 : constant
Adding op '1487' of type const
Converting op 1488 : constant
Adding op '1488' of type const
Converting op 1489 : constant
Adding op '1489' of type const
Converting op 1490 : constant
Adding op '1490' of type const
Converting op 1494 : listconstruct
Adding op '1494' of type const
Converting op 1495 : listconstruct
Adding op '1495' of type const
Converting op 1496 : listconstruct
Adding op '1496' of type const
Converting op 1497 : listconstruct
Adding op '1497' of type const
Converting op input.155 : _convolution
Adding op 'input.155' of type conv
Adding op 'input.155_pad_type_0' of type const
Adding op 'input.155_pad_0' of type const
Converting op input.163 : silu_
Adding op 'input.163' of type silu
Converting op 1500 : constant
Adding op '1500' of type const
Converting op 1501 : constant
Adding op '1501' of type const
Converting op 1502 : constant
Adding op '1502' of type const
Converting op 1503 : constant
Adding op '1503' of type const
Converting op 1507 : listconstruct
Adding op '1507' of type const
Converting op 1508 : listconstruct
Adding op '1508' of type const
Converting op 1509 : listconstruct
Adding op '1509' of type const
Converting op 1510 : listconstruct
Adding op '1510' of type const
Converting op input.157 : _convolution
Adding op 'input.157' of type conv
Adding op 'input.157_pad_type_0' of type const
Adding op 'input.157_pad_0' of type const
Converting op input.165 : silu_
Adding op 'input.165' of type silu
Converting op 1513 : constant
Adding op '1513' of type const
Converting op 1514 : constant
Adding op '1514' of type const
Converting op 1515 : constant
Adding op '1515' of type const
Converting op 1516 : constant
Adding op '1516' of type const
Converting op 1520 : listconstruct
Adding op '1520' of type const
Converting op 1521 : listconstruct
Adding op '1521' of type const
Converting op 1522 : listconstruct
Adding op '1522' of type const
Converting op 1523 : listconstruct
Adding op '1523' of type const
Converting op input.161 : _convolution
Adding op 'input.161' of type conv
Adding op 'input.161_pad_type_0' of type const
Adding op 'input.161_pad_0' of type const
Converting op input : silu_
Adding op 'input' of type silu
Converting op 1526 : constant
Adding op '1526' of type const
Converting op 1527 : constant
Adding op '1527' of type const
Converting op 1528 : constant
Adding op '1528' of type const
Converting op 1529 : constant
Adding op '1529' of type const
Converting op 1530 : constant
Adding op '1530' of type const
Converting op 1531 : constant
Adding op '1531' of type const
Converting op 1532 : constant
Adding op '1532' of type const
Converting op 1533 : constant
Adding op '1533' of type const
Converting op 1534 : constant
Adding op '1534' of type const
Converting op 1535 : constant
Adding op '1535' of type const
Converting op 1536 : constant
Adding op '1536' of type const
Converting op 1537 : constant
Adding op '1537' of type const
Converting op 1538 : constant
Adding op '1538' of type const
Converting op 1548 : listconstruct
Adding op '1548' of type const
Converting op 1549 : listconstruct
Adding op '1549' of type const
Converting op 1550 : listconstruct
Adding op '1550' of type const
Converting op 1551 : listconstruct
Adding op '1551' of type const
Converting op 1552 : _convolution
Adding op '1552' of type conv
Adding op '1552_pad_type_0' of type const
Adding op '1552_pad_0' of type const
Converting op 1553 : size
Adding op '1553_shape' of type shape
Adding op 'const_0' of type const
Converting op 1554 : size
Adding op '1554_shape' of type shape
Adding op 'const_1' of type const
Converting op 1555 : size
Adding op '1555_shape' of type shape
Adding op 'const_2' of type const
Converting op 1556 : listconstruct
Adding op '1556' of type const
Converting op 1557 : view
Adding op 'cast_0' of type cast
Adding op 'cast_0_dtype_0' of type const
Adding op '1557' of type reshape
Converting op 1558 : listconstruct
Adding op '1558' of type const
Converting op 1559 : permute
Adding op '1559' of type transpose
Converting op 1560 : contiguous
Setting pytorch op:   %1560 = contiguous[](%1559, %1536) to no-op.
Converting op y.1 : sigmoid
Adding op 'y.1' of type sigmoid
Converting op 1562 : slice
Adding op '1562' of type slice_by_index
Adding op '1562_begin_0' of type const
Adding op '1562_end_0' of type const
Adding op '1562_end_mask_0' of type const
Converting op 1563 : constant
Adding op '1563' of type const
Converting op 1564 : mul
Adding op '1564' of type mul
Converting op 1565 : constant
Adding op '1565' of type const
Converting op 1566 : sub
Adding op '1566' of type sub
Converting op 1567 : add
Adding op '1567' of type add
Converting op 1568 : select
Adding op '1568' of type slice_by_index
Adding op '1568_begin_0' of type const
Adding op '1568_end_0' of type const
Adding op '1568_end_mask_0' of type const
Adding op '1568_squeeze_mask_0' of type const
Converting op 1569 : mul
Adding op '1569' of type mul
Converting op 1570 : slice
Adding op '1570' of type slice_by_index
Adding op '1570_begin_0' of type const
Adding op '1570_end_0' of type const
Adding op '1570_end_mask_0' of type const
Converting op y.1_internal_tensor_assign_1 : _internal_op_tensor_inplace_copy_
Adding op 'expand_dims_0' of type expand_dims
Adding op 'expand_dims_0_axes_0' of type const
Adding op 'expand_dims_1' of type expand_dims
Adding op 'expand_dims_1_axes_0' of type const
Adding op 'concat_0' of type concat
Adding op 'concat_0_values1_0' of type const
Adding op 'concat_0_values2_0' of type const
Adding op 'concat_0_values3_0' of type const
Adding op 'concat_0_values4_0' of type const
Adding op 'concat_0_axis_0' of type const
Adding op 'concat_0_interleave_0' of type const
Adding op 'concat_1' of type concat
Adding op 'concat_1_values1_0' of type const
Adding op 'concat_1_values2_0' of type const
Adding op 'concat_1_values3_0' of type const
Adding op 'concat_1_values4_0' of type const
Adding op 'concat_1_axis_0' of type const
Adding op 'concat_1_interleave_0' of type const
Adding op 'y.1_internal_tensor_assign_1' of type torch_tensor_assign
Adding op 'y.1_internal_tensor_assign_1_stride_0' of type const
Adding op 'y.1_internal_tensor_assign_1_begin_mask_0' of type const
Adding op 'y.1_internal_tensor_assign_1_end_mask_0' of type const
Adding op 'y.1_internal_tensor_assign_1_squeeze_mask_0' of type const
Converting PyTorch Frontend ==> MIL Ops:  93%|██████████████████████████████▋  | 1118/1203 [00:00<00:00, 2167.55 ops/s]
CoreML export failure: The updates tensor should have shape [1, 3, 80, 80, 85]. Got (1, 3, 80, 80, 2)

Starting TorchScript-Lite export with torch 2.1.2+cu121...
TorchScript-Lite export success, saved as ./yolov7.torchscript.ptl

Starting ONNX export with onnx 1.15.0...

Starting export end2end onnx model for TensorRT...
C:\Users\User\anaconda3\envs\tensorrt\lib\site-packages\torch\nn\modules\module.py:844: UserWarning: The .grad attribute of a Tensor that is not a leaf Tensor is being accessed. Its .grad attribute won't be populated during autograd.backward(). If you indeed want the .grad field to be populated for a non-leaf Tensor, use .retain_grad() on the non-leaf Tensor. If you access the non-leaf Tensor by mistake, make sure you access the leaf Tensor instead. See github.com/pytorch/pytorch/pull/30531 for more informations. (Triggered internally at aten\src\ATen/core/TensorBody.h:494.)
  if param.grad is not None:
[W shape_type_inference.cpp:1978] Warning: The shape inference of TRT::EfficientNMS_TRT type is missing, so it may result in wrong shape inference for the exported graph. Please consider adding it in symbolic function. (function UpdateReliable)
[W shape_type_inference.cpp:1978] Warning: The shape inference of TRT::EfficientNMS_TRT type is missing, so it may result in wrong shape inference for the exported graph. Please consider adding it in symbolic function. (function UpdateReliable)
[W shape_type_inference.cpp:1978] Warning: The shape inference of TRT::EfficientNMS_TRT type is missing, so it may result in wrong shape inference for the exported graph. Please consider adding it in symbolic function. (function UpdateReliable)
[W shape_type_inference.cpp:1978] Warning: The shape inference of TRT::EfficientNMS_TRT type is missing, so it may result in wrong shape inference for the exported graph. Please consider adding it in symbolic function. (function UpdateReliable)
Simplifier failure: No module named 'onnxsim'
ONNX export success, saved as ./yolov7.onnx

Export complete (21.89s). Visualize with https://github.com/lutzroeder/netron.

(tensorrt) C:\Users\User\Downloads\yolov7-TensorRT>git clone https://github.com/Linaom1214/tensorrt-python.git
Cloning into 'tensorrt-python'...
remote: Enumerating objects: 337, done.
remote: Counting objects: 100% (83/83), done.
remote: Compressing objects: 100% (26/26), done.
remote: Total 337 (delta 61), reused 57 (delta 57), pack-reused 254Receiving objects:  94% (317/337), 129.09 MiB | 10.62Receiving objects:  97% (327/337), 129.09 MiB | 10.62 MiB/s
Receiving objects: 100% (337/337), 132.56 MiB | 8.07 MiB/s, done.
Resolving deltas: 100% (169/169), done.

(tensorrt) C:\Users\User\Downloads\yolov7-TensorRT>python tensorrt-python\export.py -o yolov7.onnx -e yolov7-nms.trt -p
fp16
Traceback (most recent call last):
  File "C:\Users\User\Downloads\yolov7-TensorRT\tensorrt-python\export.py", line 8, in <module>
    import pycuda.driver as cuda
ModuleNotFoundError: No module named 'pycuda'

(tensorrt) C:\Users\User\Downloads\yolov7-TensorRT>pip install pycuda
Collecting pycuda
  Downloading pycuda-2024.1.tar.gz (1.7 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.7/1.7 MB 2.9 MB/s eta 0:00:00
  Installing build dependencies ... done
  Getting requirements to build wheel ... done
  Preparing metadata (pyproject.toml) ... done
Collecting pytools>=2011.2 (from pycuda)
  Downloading pytools-2023.1.1-py2.py3-none-any.whl.metadata (2.7 kB)
Collecting appdirs>=1.4.0 (from pycuda)
  Downloading appdirs-1.4.4-py2.py3-none-any.whl (9.6 kB)
Collecting mako (from pycuda)
  Using cached Mako-1.3.0-py3-none-any.whl.metadata (2.9 kB)
Collecting platformdirs>=2.2.0 (from pytools>=2011.2->pycuda)
  Using cached platformdirs-4.1.0-py3-none-any.whl.metadata (11 kB)
Requirement already satisfied: typing-extensions>=4.0 in c:\users\user\anaconda3\envs\tensorrt\lib\site-packages (from pytools>=2011.2->pycuda) (4.9.0)
Requirement already satisfied: MarkupSafe>=0.9.2 in c:\users\user\anaconda3\envs\tensorrt\lib\site-packages (from mako->pycuda) (2.1.3)
Downloading pytools-2023.1.1-py2.py3-none-any.whl (70 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 70.6/70.6 kB ? eta 0:00:00
Using cached Mako-1.3.0-py3-none-any.whl (78 kB)
Using cached platformdirs-4.1.0-py3-none-any.whl (17 kB)
Building wheels for collected packages: pycuda
  Building wheel for pycuda (pyproject.toml) ... done
  Created wheel for pycuda: filename=pycuda-2024.1-cp310-cp310-win_amd64.whl size=370727 sha256=892f724229a43fcd3ca0785cdfcdbbfb7d15ea412d0afcf48f3726c93e8b862c
  Stored in directory: c:\users\user\appdata\local\pip\cache\wheels\12\34\d2\9a349255a4eca3a486d82c79d21e138ce2ccd90f414d9d72b8
Successfully built pycuda
Installing collected packages: appdirs, platformdirs, mako, pytools, pycuda
Successfully installed appdirs-1.4.4 mako-1.3.0 platformdirs-4.1.0 pycuda-2024.1 pytools-2023.1.1

(tensorrt) C:\Users\User\Downloads\yolov7-TensorRT>python tensorrt-python\export.py -o yolov7.onnx -e yolov7-nms.trt -p fp16
Namespace(onnx='yolov7.onnx', engine='yolov7-nms.trt', precision='fp16', verbose=False, workspace=1, calib_input=None, calib_cache='./calibration.cache', calib_num_images=5000, calib_batch_size=8, end2end=False, conf_thres=0.4, iou_thres=0.5, max_det=100, v8=False)
[01/09/2024-14:48:29] [TRT] [I] [MemUsageChange] Init CUDA: CPU +0, GPU +0, now: CPU 8112, GPU 1033 (MiB)
[01/09/2024-14:49:01] [TRT] [I] [MemUsageChange] Init builder kernel library: CPU +856, GPU +172, now: CPU 10162, GPU 1205 (MiB)
[01/09/2024-14:49:01] [TRT] [W] onnx2trt_utils.cpp:374: Your ONNX model has been generated with INT64 weights, while TensorRT does not natively support INT64. Attempting to cast down to INT32.
[01/09/2024-14:49:01] [TRT] [W] onnx2trt_utils.cpp:400: One or more weights outside the range of INT32 was clamped
[01/09/2024-14:49:01] [TRT] [I] No importer registered for op: EfficientNMS_TRT. Attempting to import as plugin.
[01/09/2024-14:49:01] [TRT] [I] Searching for plugin: EfficientNMS_TRT, plugin_version: 1, plugin_namespace:
[01/09/2024-14:49:01] [TRT] [W] builtin_op_importers.cpp:5221: Attribute class_agnostic not found in plugin node! Ensure that the plugin creator has a default value defined or the engine may fail to build.
[01/09/2024-14:49:01] [TRT] [I] Successfully created plugin: EfficientNMS_TRT
Network Description
Input 'images' with shape (1, 3, 640, 640) and dtype DataType.FLOAT
Output 'num_dets' with shape (1, 1) and dtype DataType.INT32
Output 'det_boxes' with shape (1, 100, 4) and dtype DataType.FLOAT
Output 'det_scores' with shape (1, 100) and dtype DataType.FLOAT
Output 'det_classes' with shape (1, 100) and dtype DataType.INT32
Building fp16 Engine in C:\Users\User\Downloads\yolov7-TensorRT\yolov7-nms.trt
[01/09/2024-14:49:01] [TRT] [I] BuilderFlag::kTF32 is set but hardware does not support TF32. Disabling TF32.
[01/09/2024-14:49:01] [TRT] [I] Graph optimization time: 0.107504 seconds.
[01/09/2024-14:49:01] [TRT] [I] [MemUsageChange] Init cuBLAS/cuBLASLt: CPU +3, GPU +10, now: CPU 9299, GPU 1215 (MiB)
[01/09/2024-14:49:03] [TRT] [I] [MemUsageChange] Init cuDNN: CPU -6, GPU +8, now: CPU 9294, GPU 1223 (MiB)
[01/09/2024-14:49:03] [TRT] [I] BuilderFlag::kTF32 is set but hardware does not support TF32. Disabling TF32.
[01/09/2024-14:49:03] [TRT] [I] Local timing cache in use. Profiling results in this builder pass will not be stored.
[01/09/2024-14:57:03] [TRT] [I] Detected 1 inputs and 4 output network tensors.
[01/09/2024-14:57:03] [TRT] [I] Total Host Persistent Memory: 409520
[01/09/2024-14:57:03] [TRT] [I] Total Device Persistent Memory: 3017728
[01/09/2024-14:57:03] [TRT] [I] Total Scratch Memory: 40320768
[01/09/2024-14:57:03] [TRT] [I] [MemUsageStats] Peak memory usage of TRT CPU/GPU memory allocators: CPU 95 MiB, GPU 770 MiB
[01/09/2024-14:57:03] [TRT] [I] [BlockAssignment] Started assigning block shifts. This will take 156 steps to complete.
[01/09/2024-14:57:03] [TRT] [I] [BlockAssignment] Algorithm ShiftNTopDown took 14.102ms to assign 12 blocks to 156 nodes requiring 73910784 bytes.
[01/09/2024-14:57:03] [TRT] [I] Total Activation Memory: 73908224
[01/09/2024-14:57:04] [TRT] [I] [MemUsageChange] Init cuBLAS/cuBLASLt: CPU +0, GPU +8, now: CPU 8998, GPU 1349 (MiB)
[01/09/2024-14:57:04] [TRT] [I] [MemUsageChange] Init cuDNN: CPU +0, GPU +10, now: CPU 8998, GPU 1359 (MiB)
[01/09/2024-14:57:04] [TRT] [W] TensorRT encountered issues when converting weights between types and that could affect accuracy.
[01/09/2024-14:57:04] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to adjust the magnitude of the weights.
[01/09/2024-14:57:04] [TRT] [W] Check verbose logs for the list of affected weights.
[01/09/2024-14:57:04] [TRT] [W] - 82 weights are affected by this issue: Detected subnormal FP16 values.
[01/09/2024-14:57:04] [TRT] [W] - 2 weights are affected by this issue: Detected values less than smallest positive FP16 subnormal value and converted them to the FP16 minimum subnormalized value.
[01/09/2024-14:57:04] [TRT] [I] [MemUsageChange] TensorRT-managed allocation in building engine: CPU +70, GPU +73, now: CPU 70, GPU 73 (MiB)
Serializing engine to file: C:\Users\User\Downloads\yolov7-TensorRT\yolov7-nms.trt

(tensorrt) C:\Users\User\Downloads\yolov7-TensorRT>python detect_tensorrt.py
[01/09/2024-15:10:19] [TRT] [I] Loaded engine size: 72 MiB
[01/09/2024-15:10:20] [TRT] [I] [MemUsageChange] Init cuBLAS/cuBLASLt: CPU +13, GPU +10, now: CPU 8034, GPU 1117 (MiB)
[01/09/2024-15:10:20] [TRT] [I] [MemUsageChange] Init cuDNN: CPU +0, GPU +8, now: CPU 8034, GPU 1125 (MiB)
[01/09/2024-15:10:20] [TRT] [W] TensorRT was linked against cuDNN 8.9.0 but loaded cuDNN 8.8.1
[01/09/2024-15:10:20] [TRT] [I] [MemUsageChange] TensorRT-managed allocation in engine deserialization: CPU +0, GPU +73, now: CPU 0, GPU 73 (MiB)
C:\Users\User\Downloads\yolov7-TensorRT\detect_tensorrt.py:23: DeprecationWarning: Use get_tensor_name instead.
  name = model.get_binding_name(index)
C:\Users\User\Downloads\yolov7-TensorRT\detect_tensorrt.py:24: DeprecationWarning: Use get_tensor_dtype instead.
  dtype = trt.nptype(model.get_binding_dtype(index))
C:\Users\User\Downloads\yolov7-TensorRT\detect_tensorrt.py:25: DeprecationWarning: Use get_tensor_shape instead.
  shape = tuple(model.get_binding_shape(index))
[01/09/2024-15:10:20] [TRT] [I] [MemUsageChange] Init cuBLAS/cuBLASLt: CPU +0, GPU +8, now: CPU 7970, GPU 1141 (MiB)
[01/09/2024-15:10:20] [TRT] [I] [MemUsageChange] Init cuDNN: CPU +2, GPU +8, now: CPU 7972, GPU 1149 (MiB)
[01/09/2024-15:10:20] [TRT] [W] TensorRT was linked against cuDNN 8.9.0 but loaded cuDNN 8.8.1
[01/09/2024-15:10:20] [TRT] [I] [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +0, GPU +73, now: CPU 0, GPU 146 (MiB)
torch.Size([1, 3, 640, 640])
Cost 0.007027099999959319 s
```
</details>
