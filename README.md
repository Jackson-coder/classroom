<div align="center">
  <p>
    <a align="center" href="https://ultralytics.com/yolov5" target="_blank">
      <img width="850" src="https://github.com/ultralytics/assets/blob/master/yolov5/v62/splash_readme.png"></a>
  </p>

  English | [简体中文](.github/README_cn.md)
  <br>
  <div>
    <a href="https://github.com/ultralytics/yolov5/actions/workflows/ci-testing.yml"><img src="https://github.com/ultralytics/yolov5/actions/workflows/ci-testing.yml/badge.svg" alt="YOLOv5 CI"></a>
    <a href="https://zenodo.org/badge/latestdoi/264818686"><img src="https://zenodo.org/badge/264818686.svg" alt="YOLOv5 Citation"></a>
    <a href="https://hub.docker.com/r/ultralytics/yolov5"><img src="https://img.shields.io/docker/pulls/ultralytics/yolov5?logo=docker" alt="Docker Pulls"></a>
    <br>
    <a href="https://bit.ly/yolov5-paperspace-notebook"><img src="https://assets.paperspace.io/img/gradient-badge.svg" alt="Run on Gradient"></a>
    <a href="https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
    <a href="https://www.kaggle.com/ultralytics/yolov5"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle"></a>
  </div>

  <br>
  <p>
    YOLOv5 🚀 is the world's most loved vision AI, representing <a href="https://ultralytics.com">Ultralytics</a>
    open-source research into future vision AI methods, incorporating lessons learned and best practices evolved over thousands of hours of research and development. To request a commercial license please complete the form at <a href="https://ultralytics.com/license">Ultralytics Licensing</a>.

  </p>

  <div align="center">
    <a href="https://github.com/ultralytics" style="text-decoration:none;">
      <img src="https://github.com/ultralytics/assets/raw/master/social/logo-social-github.png" width="2%" alt="" /></a>
    <img src="https://github.com/ultralytics/assets/raw/master/social/logo-transparent.png" width="2%" alt="" />
    <a href="https://www.linkedin.com/company/ultralytics" style="text-decoration:none;">
      <img src="https://github.com/ultralytics/assets/raw/master/social/logo-social-linkedin.png" width="2%" alt="" /></a>
    <img src="https://github.com/ultralytics/assets/raw/master/social/logo-transparent.png" width="2%" alt="" />
    <a href="https://twitter.com/ultralytics" style="text-decoration:none;">
      <img src="https://github.com/ultralytics/assets/raw/master/social/logo-social-twitter.png" width="2%" alt="" /></a>
    <img src="https://github.com/ultralytics/assets/raw/master/social/logo-transparent.png" width="2%" alt="" />
    <a href="https://www.producthunt.com/@glenn_jocher" style="text-decoration:none;">
      <img src="https://github.com/ultralytics/assets/raw/master/social/logo-social-producthunt.png" width="2%" alt="" /></a>
    <img src="https://github.com/ultralytics/assets/raw/master/social/logo-transparent.png" width="2%" alt="" />
    <a href="https://youtube.com/ultralytics" style="text-decoration:none;">
      <img src="https://github.com/ultralytics/assets/raw/master/social/logo-social-youtube.png" width="2%" alt="" /></a>
    <img src="https://github.com/ultralytics/assets/raw/master/social/logo-transparent.png" width="2%" alt="" />
    <a href="https://www.facebook.com/ultralytics" style="text-decoration:none;">
      <img src="https://github.com/ultralytics/assets/raw/master/social/logo-social-facebook.png" width="2%" alt="" /></a>
    <img src="https://github.com/ultralytics/assets/raw/master/social/logo-transparent.png" width="2%" alt="" />
    <a href="https://www.instagram.com/ultralytics/" style="text-decoration:none;">
      <img src="https://github.com/ultralytics/assets/raw/master/social/logo-social-instagram.png" width="2%" alt="" /></a>
  </div>
</div>


## <div align="center">Documentation</div>

See the [YOLOv5 Docs](https://docs.ultralytics.com) for full documentation on training, testing and deployment.

## <div align="center">Quick Start Examples</div>

<details open>
<summary>Install</summary>

Clone repo and install [requirements.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt) in a
[**Python>=3.7.0**](https://www.python.org/) environment, including
[**PyTorch>=1.7**](https://pytorch.org/get-started/locally/).

```bash
git clone https://github.com/Jackson-coder/classroom  # clone
cd classroom
pip install -r requirements.txt  # install
```
### Get simsun.zip via the following link and unzip it in the root directory.

link：https://pan.baidu.com/s/14WMopYlf4pdOTpNcgLfKPg?pwd=0000 

password：0000 


</details>

<details>
<summary>Inference with detect.py</summary>

`detect.py` runs inference on a variety of sources, downloading [models](https://github.com/ultralytics/yolov5/tree/master/models) automatically from
the latest YOLOv5 [release](https://github.com/ultralytics/yolov5/releases) and saving results to `runs/detect`.

```bash
python detect.py --weights runs/train/exp2/weights/best.pt --source data/videos --data school.yaml --img 960 --device 2
```

</details>

<details>
<summary>Training</summary>

```bash
python train.py --img 960 --batch 4 --workers 4 --device 2 --epochs 300 --cfg yolov5l.yaml --data school.yaml --weights yolov5l.pt
```
</details>



## <div align="center">Integrations</div>

<img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/image-integrations-loop.png" width="100%" />

<div align="center">
  <a href="https://roboflow.com/?ref=ultralytics">
    <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-roboflow.png" width="10%" /></a>
  <img src="https://github.com/ultralytics/assets/raw/master/social/logo-transparent.png" width="15%" height="0" alt="" />
  <a href="https://cutt.ly/yolov5-readme-clearml">
    <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-clearml.png" width="10%" /></a>
  <img src="https://github.com/ultralytics/assets/raw/master/social/logo-transparent.png" width="15%" height="0" alt="" />
  <a href="https://bit.ly/yolov5-readme-comet">
    <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-comet.png" width="10%" /></a>
  <img src="https://github.com/ultralytics/assets/raw/master/social/logo-transparent.png" width="15%" height="0" alt="" />
  <a href="https://bit.ly/yolov5-deci-platform">
    <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-deci.png" width="10%" /></a>
</div>

|Roboflow|ClearML ⭐ NEW|Comet ⭐ NEW|Deci ⭐ NEW|
|:-:|:-:|:-:|:-:|
|Label and export your custom datasets directly to YOLOv5 for training with [Roboflow](https://roboflow.com/?ref=ultralytics)|Automatically track, visualize and even remotely train YOLOv5 using [ClearML](https://cutt.ly/yolov5-readme-clearml) (open-source!)|Free forever, [Comet](https://bit.ly/yolov5-readme-comet) lets you save YOLOv5 models, resume training, and interactively visualise and debug predictions|Automatically compile and quantize YOLOv5 for better inference performance in one click at [Deci](https://bit.ly/yolov5-deci-platform)|





<details>
  <summary>Table Notes (click to expand)</summary>

- All checkpoints are trained to 300 epochs with default settings. Nano and Small models use [hyp.scratch-low.yaml](https://github.com/ultralytics/yolov5/blob/master/data/hyps/hyp.scratch-low.yaml) hyps, all others use [hyp.scratch-high.yaml](https://github.com/ultralytics/yolov5/blob/master/data/hyps/hyp.scratch-high.yaml).
- **mAP<sup>val</sup>** values are for single-model single-scale on [COCO val2017](http://cocodataset.org) dataset.<br>Reproduce by `python val.py --data coco.yaml --img 640 --conf 0.001 --iou 0.65`
- **Speed** averaged over COCO val images using a [AWS p3.2xlarge](https://aws.amazon.com/ec2/instance-types/p3/) instance. NMS times (~1 ms/img) not included.<br>Reproduce by `python val.py --data coco.yaml --img 640 --task speed --batch 1`
- **TTA** [Test Time Augmentation](https://github.com/ultralytics/yolov5/issues/303) includes reflection and scale augmentations.<br>Reproduce by `python val.py --data coco.yaml --img 1536 --iou 0.7 --augment`

</details>

## <div align="center">Classification ⭐ NEW</div>

YOLOv5 [release v6.2](https://github.com/ultralytics/yolov5/releases) brings support for classification model training, validation, prediction and export! We've made training classifier models super simple. Click below to get started.

<details>
  <summary>Classification Checkpoints (click to expand)</summary>

<br>

We trained YOLOv5-cls classification models on ImageNet for 90 epochs using a 4xA100 instance, and we trained ResNet and EfficientNet models alongside with the same default training settings to compare. We exported all models to ONNX FP32 for CPU speed tests and to TensorRT FP16 for GPU speed tests. We ran all speed tests on Google [Colab Pro](https://colab.research.google.com/signup) for easy reproducibility.

| Model                                                                                              | size<br><sup>(pixels) | acc<br><sup>top1 | acc<br><sup>top5 | Training<br><sup>90 epochs<br>4xA100 (hours) | Speed<br><sup>ONNX CPU<br>(ms) | Speed<br><sup>TensorRT V100<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>@224 (B) |
|----------------------------------------------------------------------------------------------------|-----------------------|------------------|------------------|----------------------------------------------|--------------------------------|-------------------------------------|--------------------|------------------------|
| [YOLOv5n-cls](https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5n-cls.pt)         | 224                   | 64.6             | 85.4             | 7:59                                         | **3.3**                        | **0.5**                             | **2.5**            | **0.5**                |
| [YOLOv5s-cls](https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5s-cls.pt)         | 224                   | 71.5             | 90.2             | 8:09                                         | 6.6                            | 0.6                                 | 5.4                | 1.4                    |
| [YOLOv5m-cls](https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5m-cls.pt)         | 224                   | 75.9             | 92.9             | 10:06                                        | 15.5                           | 0.9                                 | 12.9               | 3.9                    |
| [YOLOv5l-cls](https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5l-cls.pt)         | 224                   | 78.0             | 94.0             | 11:56                                        | 26.9                           | 1.4                                 | 26.5               | 8.5                    |
| [YOLOv5x-cls](https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5x-cls.pt)         | 224                   | **79.0**         | **94.4**         | 15:04                                        | 54.3                           | 1.8                                 | 48.1               | 15.9                   |
|                                                                                                    |
| [ResNet18](https://github.com/ultralytics/yolov5/releases/download/v6.2/resnet18.pt)               | 224                   | 70.3             | 89.5             | **6:47**                                     | 11.2                           | 0.5                                 | 11.7               | 3.7                    |
| [ResNet34](https://github.com/ultralytics/yolov5/releases/download/v6.2/resnet34.pt)               | 224                   | 73.9             | 91.8             | 8:33                                         | 20.6                           | 0.9                                 | 21.8               | 7.4                    |
| [ResNet50](https://github.com/ultralytics/yolov5/releases/download/v6.2/resnet50.pt)               | 224                   | 76.8             | 93.4             | 11:10                                        | 23.4                           | 1.0                                 | 25.6               | 8.5                    |
| [ResNet101](https://github.com/ultralytics/yolov5/releases/download/v6.2/resnet101.pt)             | 224                   | 78.5             | 94.3             | 17:10                                        | 42.1                           | 1.9                                 | 44.5               | 15.9                   |
|                                                                                                    |
| [EfficientNet_b0](https://github.com/ultralytics/yolov5/releases/download/v6.2/efficientnet_b0.pt) | 224                   | 75.1             | 92.4             | 13:03                                        | 12.5                           | 1.3                                 | 5.3                | 1.0                    |
| [EfficientNet_b1](https://github.com/ultralytics/yolov5/releases/download/v6.2/efficientnet_b1.pt) | 224                   | 76.4             | 93.2             | 17:04                                        | 14.9                           | 1.6                                 | 7.8                | 1.5                    |
| [EfficientNet_b2](https://github.com/ultralytics/yolov5/releases/download/v6.2/efficientnet_b2.pt) | 224                   | 76.6             | 93.4             | 17:10                                        | 15.9                           | 1.6                                 | 9.1                | 1.7                    |
| [EfficientNet_b3](https://github.com/ultralytics/yolov5/releases/download/v6.2/efficientnet_b3.pt) | 224                   | 77.7             | 94.0             | 19:19                                        | 18.9                           | 1.9                                 | 12.2               | 2.4                    |

<details>
  <summary>Table Notes (click to expand)</summary>

- All checkpoints are trained to 90 epochs with SGD optimizer with `lr0=0.001` and `weight_decay=5e-5` at image size 224 and all default settings.<br>Runs logged to https://wandb.ai/glenn-jocher/YOLOv5-Classifier-v6-2
- **Accuracy** values are for single-model single-scale on [ImageNet-1k](https://www.image-net.org/index.php) dataset.<br>Reproduce by `python classify/val.py --data ../datasets/imagenet --img 224`
- **Speed** averaged over 100 inference images using a Google [Colab Pro](https://colab.research.google.com/signup) V100 High-RAM instance.<br>Reproduce by `python classify/val.py --data ../datasets/imagenet --img 224 --batch 1`
- **Export** to ONNX at FP32 and TensorRT at FP16 done with `export.py`. <br>Reproduce by `python export.py --weights yolov5s-cls.pt --include engine onnx --imgsz 224`
</details>
</details>

<details>
  <summary>Classification Usage Examples (click to expand)</summary>

### Train
YOLOv5 classification training supports auto-download of MNIST, Fashion-MNIST, CIFAR10, CIFAR100, Imagenette, Imagewoof, and ImageNet datasets with the `--data` argument. To start training on MNIST for example use `--data mnist`.

```bash
# Single-GPU
python classify/train.py --model yolov5s-cls.pt --data cifar100 --epochs 5 --img 224 --batch 128

# Multi-GPU DDP
python -m torch.distributed.run --nproc_per_node 4 --master_port 1 classify/train.py --model yolov5s-cls.pt --data imagenet --epochs 5 --img 224 --device 0,1,2,3
```

### Val
Validate YOLOv5m-cls accuracy on ImageNet-1k dataset:
```bash
bash data/scripts/get_imagenet.sh --val  # download ImageNet val split (6.3G, 50000 images)
python classify/val.py --weights yolov5m-cls.pt --data ../datasets/imagenet --img 224  # validate
```

### Predict
Use pretrained YOLOv5s-cls.pt to predict bus.jpg:
```bash
python classify/predict.py --weights yolov5s-cls.pt --data data/images/bus.jpg
```
```python
model = torch.hub.load('ultralytics/yolov5', 'custom', 'yolov5s-cls.pt')  # load from PyTorch Hub
```

### Export
Export a group of trained YOLOv5s-cls, ResNet and EfficientNet models to ONNX and TensorRT:
```bash
python export.py --weights yolov5s-cls.pt resnet50.pt efficientnet_b0.pt --include onnx engine --img 224
```
</details>


## <div align="center">Environments</div>

Get started in seconds with our verified environments. Click each icon below for details.

<div align="center">
  <a href="https://bit.ly/yolov5-paperspace-notebook">
    <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-gradient.png" width="10%" /></a>
  <img src="https://github.com/ultralytics/assets/raw/master/social/logo-transparent.png" width="5%" alt="" />
  <a href="https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb">
    <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-colab-small.png" width="10%" /></a>
  <img src="https://github.com/ultralytics/assets/raw/master/social/logo-transparent.png" width="5%" alt="" />
  <a href="https://www.kaggle.com/ultralytics/yolov5">
    <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-kaggle-small.png" width="10%" /></a>
  <img src="https://github.com/ultralytics/assets/raw/master/social/logo-transparent.png" width="5%" alt="" />
  <a href="https://hub.docker.com/r/ultralytics/yolov5">
    <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-docker-small.png" width="10%" /></a>
  <img src="https://github.com/ultralytics/assets/raw/master/social/logo-transparent.png" width="5%" alt="" />
  <a href="https://github.com/ultralytics/yolov5/wiki/AWS-Quickstart">
    <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-aws-small.png" width="10%" /></a>
  <img src="https://github.com/ultralytics/assets/raw/master/social/logo-transparent.png" width="5%" alt="" />
  <a href="https://github.com/ultralytics/yolov5/wiki/GCP-Quickstart">
    <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-gcp-small.png" width="10%" /></a>
</div>


## <div align="center">Contribute</div>

We love your input! We want to make contributing to YOLOv5 as easy and transparent as possible. Please see our [Contributing Guide](CONTRIBUTING.md) to get started, and fill out the [YOLOv5 Survey](https://ultralytics.com/survey?utm_source=github&utm_medium=social&utm_campaign=Survey) to send us feedback on your experiences. Thank you to all our contributors!

<!-- SVG image from https://opencollective.com/ultralytics/contributors.svg?width=990 -->
<a href="https://github.com/ultralytics/yolov5/graphs/contributors"><img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/image-contributors-1280.png" /></a>

## <div align="center">Contact</div>

For YOLOv5 bugs and feature requests please visit [GitHub Issues](https://github.com/ultralytics/yolov5/issues). For professional support please [Contact Us](https://ultralytics.com/contact). To request a commercial license please complete the form at [Ultralytics Licensing](https://ultralytics.com/license).

<br>
<div align="center">
  <a href="https://github.com/ultralytics" style="text-decoration:none;">
    <img src="https://github.com/ultralytics/assets/raw/master/social/logo-social-github.png" width="3%" alt="" /></a>
  <img src="https://github.com/ultralytics/assets/raw/master/social/logo-transparent.png" width="3%" alt="" />
  <a href="https://www.linkedin.com/company/ultralytics" style="text-decoration:none;">
    <img src="https://github.com/ultralytics/assets/raw/master/social/logo-social-linkedin.png" width="3%" alt="" /></a>
  <img src="https://github.com/ultralytics/assets/raw/master/social/logo-transparent.png" width="3%" alt="" />
  <a href="https://twitter.com/ultralytics" style="text-decoration:none;">
    <img src="https://github.com/ultralytics/assets/raw/master/social/logo-social-twitter.png" width="3%" alt="" /></a>
  <img src="https://github.com/ultralytics/assets/raw/master/social/logo-transparent.png" width="3%" alt="" />
  <a href="https://www.producthunt.com/@glenn_jocher" style="text-decoration:none;">
    <img src="https://github.com/ultralytics/assets/raw/master/social/logo-social-producthunt.png" width="3%" alt="" /></a>
  <img src="https://github.com/ultralytics/assets/raw/master/social/logo-transparent.png" width="3%" alt="" />
  <a href="https://youtube.com/ultralytics" style="text-decoration:none;">
    <img src="https://github.com/ultralytics/assets/raw/master/social/logo-social-youtube.png" width="3%" alt="" /></a>
  <img src="https://github.com/ultralytics/assets/raw/master/social/logo-transparent.png" width="3%" alt="" />
  <a href="https://www.facebook.com/ultralytics" style="text-decoration:none;">
    <img src="https://github.com/ultralytics/assets/raw/master/social/logo-social-facebook.png" width="3%" alt="" /></a>
  <img src="https://github.com/ultralytics/assets/raw/master/social/logo-transparent.png" width="3%" alt="" />
  <a href="https://www.instagram.com/ultralytics/" style="text-decoration:none;">
    <img src="https://github.com/ultralytics/assets/raw/master/social/logo-social-instagram.png" width="3%" alt="" /></a>
</div>

[assets]: https://github.com/ultralytics/yolov5/releases
[tta]: https://github.com/ultralytics/yolov5/issues/303
