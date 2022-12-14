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

