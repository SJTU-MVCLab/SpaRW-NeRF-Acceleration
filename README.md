# SpaRW: A Plug-and-Play NeRF Acceleration Extension

This is the implementation of the SpaRW algorithm from the paper: `Cicero: Addressing Algorithmic and Architectural Bottlenecks in Neural Rendering by Radiance Warping and Memory Optimizations` [Link](https://horizon-lab.org/pubs/isca24-cicero.pdf).

[![teaser](https://github.com/user-attachments/assets/7070672a-660c-4d5d-b7f8-d2f4127dbbbc)](https://youtu.be/eCiwp5VY9Qo)
(Click this figure to see the comparison between our method and the ground truth)

## What is it?

This repository implements Cicero's sparse radiance warping (SpaRW) algorithm (published on ISCA'2024). This algorithm exploits the radiance similarity across rays from nearby camera views to reduce the overall computation. 
Note that, SpaRW is not a new NeRF algorithm. Rather, it is a plug-and-play extension to *virtually* all NeRF algorithms. In this repository, we extend the [DirectVoxGO](https://github.com/sunset1995/DirectVoxGO) as an example to demonstrate our algorithm.

## How to Run

Please follow the steps to run this code.

### Installation

First, clone this repository:
```
git clone https://github.com/SJTU-MVCLab/SPARW.git
cd SPARW
pip install -r requirements.txt
```
[Pytorch](https://pytorch.org/) and [torch_scatter](https://github.com/rusty1s/pytorch_scatter) installation is machine dependent, please install the correct version for your machine.

<details>
  <summary> Dependencies (click to expand) </summary>

  - `PyTorch`, `numpy`, `torch_scatter`: main computation.
  - `scipy`, `lpips`: SSIM and LPIPS evaluation.
  - `tqdm`: progress bar.
  - `mmcv`: config system.
  - `opencv-python`: image processing.
  - `imageio`, `imageio-ffmpeg`: images and videos I/O.
  - `Ninja`: to build the newly implemented torch extention just-in-time.
  - `einops`: torch tensor shaping with pretty api.
  - `torch_efficient_distloss`: O(N) realization for the distortion loss.
</details>

### Dataset

The next step is to prepare the dataset. We use [SyntheticNeRF](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1) dataset from the original [NeRF](https://www.matthewtancik.com/nerf) algorithm with one **caveat**: *the original depth data from SyntheticNeRF is incorrect, we regenerated the depth map from original Blender files in SyntheticNeRF dataset.* To correctly warp the radiance values from the previous frames, please download the correct depth data from our dataset:
```
  https://drive.google.com/drive/folders/1svMEQ_0kdw_qQYm7wZv7RPiX2RZ6vZxJ?usp=sharing
```

The downloaded dataset should be stored in the `data` directory, the directory should be organized as follows:

```
  data
  └── nerf_synthetic    
      └── [chair|drums|hotdog|lego|mic|ship]
          ├── [train|val|test]
          │   └── r_*.png
          ├── transforms_[train|val|test].json
          |
          └── depth      # This is new compared to the original SyntheticNeRF!
              └── r_*.exr
```

### Training and Evaluation Script

Note that, our implementation **does not require training**, but we provide a training script to obtain initial model weights. To train our model:
```bash
   $ python run.py --config configs/nerf/lego.py --render_test
```

To evaluate our SpaRW algorithm, please run:
```bash
  $ python run.py --config configs/nerf/lego.py --render_only --render_test --eval_ssim
```
Please modify the configuration files in `configs/nerf` to run different scenes.

## What We Changed

We simply implement a cuda kernel `dvgo.render_utils_cuda.sparw` that can warp the image from the reference frame to the target frame image, and then use the returned mask to render the occluded areas of the image with your own model:
```python
dvgo.render_utils_cuda.sparw(ref_img, ref_depth, tgt_depth, ref_K, tgt_K, ref_c2w, tgt_c2w, H, W)
```

The detailed parameter explanation:
  - `ref_img` (`torch.Tensor`, CPU): The image in the reference frame.
  - `ref_depth` (`torch.Tensor`, GPU): The depth estimation in the reference frame.
  - `tgt_depth` (`torch.Tensor`, GPU): The depth estimation in the target frame.
  - `ref_K` (`torch.Tensor`, GPU): Reference frame intrinsic matrix tensor.
  - `tgt_K` (`torch.Tensor`, GPU): Target frame intrinsic matrix tensor.
  - `ref_c2w` (`torch.Tensor`, GPU): Reference frame camera-to-world transformation matrix tensor.
  - `tgt_c2w` (`torch.Tensor`, GPU): Target frame camera-to-world transformation matrix tensor.
  - `H` (`int`): Image height.
  - `W` (`int`): Image width.

### RETURNS:
  - `warped_img` (`torch.Tensor`, CPU): Warped image tensor. This tensor contains the image data from the reference frame that has been transformed (warped) to align with the target frame.
  - `mask` (`torch.Tensor`, GPU): Mask tensor. This tensor indicates which regions of the warped image are invalid and should be considered for further processing. Typically, it marks the occluded or invalid areas with `True`.

## Acknowledgement
The repository heavily borrows the code from the [DirectVoxGO](https://github.com/sunset1995/DirectVoxGO) implementation.

## Citation

Please kindly consider citing this paper in your publications if it helps your research.
```
@inproceedings{feng2023cicero,
  title={Cicero: Addressing Algorithmic and Architectural Bottlenecks in Neural Rendering by Radiance Warping and Memory Optimizations},
  author={Feng, Yu and Liu, Zihan and Leng, Jingwen and Guo, Minyi and Zhu, Yuhao},
  booktitle={Proceedings of the 51st Annual International Symposium on Computer Architecture},
  year={2024}
}
```
