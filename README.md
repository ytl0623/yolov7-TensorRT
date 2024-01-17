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

```
# Tested with: Windows 11(Anaconda), Python 3.10, Pytorch 2.1.2(Cuda 12.1), TensorRT 8.6.1
(tensorrt) C:\Users\User\Downloads\yolov7-TensorRT>python detect.py --weights yolov7.pt --conf 0.25 --img-size 640 --source inference/images/horses.jpg
Namespace(weights=['yolov7.pt'], source='inference/images/horses.jpg', img_size=640, conf_thres=0.25, iou_thres=0.45, device='', view_img=False, save_txt=False, save_conf=False, nosave=False, classes=None, agnostic_nms=False, augment=False, update=False, project='runs/detect', name='exp', exist_ok=False, no_trace=False)
YOLOR  d527376 torch 2.1.2+cu121 CUDA:0 (NVIDIA GeForce RTX 2060, 6143.6875MB)

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

5 horses, Done. (13.0ms) Inference, (413.1ms) NMS
 The image with the result is saved in: runs\detect\exp3\horses.jpg
Done. (0.732s)

(tensorrt) C:\Users\User\Downloads\yolov7-TensorRT>
```

<div align="center">
    <a href="./">
        <img src="./figure/horses_prediction.jpg" width="59%"/>
    </a>
</div>


## Export

**Pytorch to CoreML (and inference on MacOS/iOS)** <a href="https://colab.research.google.com/github/WongKinYiu/yolov7/blob/main/tools/YOLOv7CoreML.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>

**Pytorch to ONNX with NMS (and inference)** <a href="https://colab.research.google.com/github/WongKinYiu/yolov7/blob/main/tools/YOLOv7onnx.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
``` shell
python export.py --weights yolov7-tiny.pt --grid --end2end --simplify \
        --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size 640 640 --max-wh 640
```

**Pytorch to TensorRT with NMS (and inference)** <a href="https://colab.research.google.com/github/WongKinYiu/yolov7/blob/main/tools/YOLOv7trt.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>

``` shell
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt
python export.py --weights ./yolov7-tiny.pt --grid --end2end --simplify --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size 640 640
git clone https://github.com/Linaom1214/tensorrt-python.git
python ./tensorrt-python/export.py -o yolov7-tiny.onnx -e yolov7-tiny-nms.trt -p fp16
```

```
# Tested with: Windows 11(Anaconda), Python 3.10, Pytorch 2.1.2(Cuda 12.1), TensorRT 8.6.1
(tensorrt) C:\Users\User\Downloads\yolov7-TensorRT>python detect_tensorrt.py
[01/17/2024-14:27:52] [TRT] [I] Loaded engine size: 72 MiB
[01/17/2024-14:27:52] [TRT] [I] [MemUsageChange] Init cuBLAS/cuBLASLt: CPU +11, GPU +10, now: CPU 13092, GPU 1117 (MiB)
[01/17/2024-14:27:52] [TRT] [I] [MemUsageChange] Init cuDNN: CPU +4, GPU +8, now: CPU 13096, GPU 1125 (MiB)
[01/17/2024-14:27:52] [TRT] [W] TensorRT was linked against cuDNN 8.9.0 but loaded cuDNN 8.8.1
[01/17/2024-14:27:52] [TRT] [I] [MemUsageChange] TensorRT-managed allocation in engine deserialization: CPU +0, GPU +73, now: CPU 0, GPU 73 (MiB)
C:\Users\User\Downloads\yolov7-TensorRT\detect_tensorrt.py:25: DeprecationWarning: Use get_tensor_name instead.
  name = model.get_binding_name(index)
C:\Users\User\Downloads\yolov7-TensorRT\detect_tensorrt.py:26: DeprecationWarning: Use get_tensor_dtype instead.
  dtype = trt.nptype(model.get_binding_dtype(index))
C:\Users\User\Downloads\yolov7-TensorRT\detect_tensorrt.py:27: DeprecationWarning: Use get_tensor_shape instead.
  shape = tuple(model.get_binding_shape(index))
[01/17/2024-14:27:52] [TRT] [I] [MemUsageChange] Init cuBLAS/cuBLASLt: CPU +0, GPU +8, now: CPU 13033, GPU 1141 (MiB)
[01/17/2024-14:27:52] [TRT] [I] [MemUsageChange] Init cuDNN: CPU +2, GPU +8, now: CPU 13035, GPU 1149 (MiB)
[01/17/2024-14:27:52] [TRT] [W] TensorRT was linked against cuDNN 8.9.0 but loaded cuDNN 8.8.1
[01/17/2024-14:27:52] [TRT] [I] [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +0, GPU +73, now: CPU 0, GPU 146 (MiB)
Cost 0.3626005999976769 s

(tensorrt) C:\Users\User\Downloads\yolov7-TensorRT>
```

```
# yolov7-seg export error
(tensorrt) C:\Users\User\Downloads\yolov7-TensorRT>python export.py --weights ./yolov7-seg.pt --grid --end2end --simplify --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size 640 640
Namespace(weights='./yolov7-seg.pt', img_size=[640, 640], batch_size=1, dynamic=False, dynamic_batch=False, grid=True, end2end=True, max_wh=None, topk_all=100, iou_thres=0.65, conf_thres=0.35, device='cpu', simplify=True, include_nms=False, fp16=False, int8=False)
YOLOR  593e77d torch 2.1.2+cu121 CPU

Traceback (most recent call last):
  File "C:\Users\User\Downloads\yolov7-TensorRT\export.py", line 47, in <module>
    model = attempt_load(opt.weights, map_location=device)  # load FP32 model
  File "C:\Users\User\Downloads\yolov7-TensorRT\models\experimental.py", line 252, in attempt_load
    ckpt = torch.load(w, map_location=map_location)  # load
  File "C:\Users\User\anaconda3\envs\tensorrt\lib\site-packages\torch\serialization.py", line 1014, in load
    return _load(opened_zipfile,
  File "C:\Users\User\anaconda3\envs\tensorrt\lib\site-packages\torch\serialization.py", line 1422, in _load
    result = unpickler.load()
  File "C:\Users\User\anaconda3\envs\tensorrt\lib\site-packages\torch\serialization.py", line 1415, in find_class
    return super().find_class(mod_name, name)
AttributeError: Can't get attribute 'SegmentationModel' on <module 'models.yolo' from 'C:\\Users\\User\\Downloads\\yolov7-TensorRT\\models\\yolo.py'>

(tensorrt) C:\Users\User\Downloads\yolov7-TensorRT>
```

```
# yolov7-mask export error
(tensorrt) C:\Users\User\Downloads\yolov7-TensorRT>python export.py --weights ./yolov7-mask.pt --grid --end2end --simplify --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size 640 640
Namespace(weights='./yolov7-mask.pt', img_size=[640, 640], batch_size=1, dynamic=False, dynamic_batch=False, grid=True, end2end=True, max_wh=None, topk_all=100, iou_thres=0.65, conf_thres=0.35, device='cpu', simplify=True, include_nms=False, fp16=False, int8=False)
YOLOR  593e77d torch 2.1.2+cu121 CPU

Traceback (most recent call last):
  File "C:\Users\User\Downloads\yolov7-TensorRT\export.py", line 47, in <module>
    model = attempt_load(opt.weights, map_location=device)  # load FP32 model
  File "C:\Users\User\Downloads\yolov7-TensorRT\models\experimental.py", line 252, in attempt_load
    ckpt = torch.load(w, map_location=map_location)  # load
  File "C:\Users\User\anaconda3\envs\tensorrt\lib\site-packages\torch\serialization.py", line 1014, in load
    return _load(opened_zipfile,
  File "C:\Users\User\anaconda3\envs\tensorrt\lib\site-packages\torch\serialization.py", line 1422, in _load
    result = unpickler.load()
  File "C:\Users\User\anaconda3\envs\tensorrt\lib\site-packages\torch\serialization.py", line 1415, in find_class
    return super().find_class(mod_name, name)
AttributeError: Can't get attribute 'Merge' on <module 'models.common' from 'C:\\Users\\User\\Downloads\\yolov7-TensorRT\\models\\common.py'>

(tensorrt) C:\Users\User\Downloads\yolov7-TensorRT>
```

**Pytorch to TensorRT another way** <a href="https://colab.research.google.com/gist/AlexeyAB/fcb47ae544cf284eb24d8ad8e880d45c/yolov7trtlinaom.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a> <details><summary> <b>Expand</b> </summary>


``` shell
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

## Error
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

* 20240109
```
(tensorrt) C:\Users\User\Downloads\yolov7-TensorRT>python detect_onnx.py
Traceback (most recent call last):
  File "C:\Users\User\Downloads\yolov7-TensorRT\detect_onnx.py", line 18, in <module>
    session = ort.InferenceSession(w, providers=providers)
  File "C:\Users\User\anaconda3\envs\tensorrt\lib\site-packages\onnxruntime\capi\onnxruntime_inference_collection.py", line 419, in __init__
    self._create_inference_session(providers, provider_options, disabled_optimizers)
  File "C:\Users\User\anaconda3\envs\tensorrt\lib\site-packages\onnxruntime\capi\onnxruntime_inference_collection.py", line 452, in _create_inference_session
    sess = C.InferenceSession(session_options, self._model_path, True, self._read_config_from_model)
onnxruntime.capi.onnxruntime_pybind11_state.Fail: [ONNXRuntimeError] : 1 : FAIL : Load model from ./yolov7.onnx failed:Fatal error: TRT:EfficientNMS_TRT(-1) is not a registered function/op

(tensorrt) C:\Users\User\Downloads\yolov7-TensorRT>
```

</details>
