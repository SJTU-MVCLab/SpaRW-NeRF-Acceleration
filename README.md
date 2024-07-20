# SPARW
**This is the implementation of the SPARW algorithm from the paper [Cicero: Addressing Algorithmic and Architectural Bottlenecks in Neural Rendering by Radiance Warping and Memory Optimizations](https://arxiv.org/abs/2404.11852).**

https://github.com/user-attachments/assets/6c7dbc00-aedf-4e68-9f8c-23f368a12b24

### Installation
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


## Directory Structure For The Datasets

<details>
  <summary> (click to expand;) </summary>

    data
    ├── nerf_synthetic     # Link: https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1
    │   └── [chair|drums|ficus|hotdog|lego|materials|mic|ship]
    │       ├── [train|val|test]
    │       │   └── r_*.png
    │       ├── transforms_[train|val|test].json
    |       |
    |       └── depth
    |           └── r_*.exr
    │
    ├── Synthetic_NSVF     # Link: https://dl.fbaipublicfiles.com/nsvf/dataset/Synthetic_NSVF.zip
    │   └── [Bike|Lifestyle|Palace|Robot|Spaceship|Steamtrain|Toad|Wineholder]
    │       ├── intrinsics.txt
    │       ├── rgb
    │       │   └── [0_train|1_val|2_test]_*.png
    │       └── pose
    │           └── [0_train|1_val|2_test]_*.txt
    │
    ├── BlendedMVS         # Link: https://dl.fbaipublicfiles.com/nsvf/dataset/BlendedMVS.zip
    │   └── [Character|Fountain|Jade|Statues]
    │       ├── intrinsics.txt
    │       ├── rgb
    │       │   └── [0|1|2]_*.png
    │       └── pose
    │           └── [0|1|2]_*.txt
    │
    ├── TanksAndTemple     # Link: https://dl.fbaipublicfiles.com/nsvf/dataset/TanksAndTemple.zip
    │   └── [Barn|Caterpillar|Family|Ignatius|Truck]
    │       ├── intrinsics.txt
    │       ├── rgb
    │       │   └── [0|1|2]_*.png
    │       └── pose
    │           └── [0|1|2]_*.txt
    │
    ├── deepvoxels         # Link: https://drive.google.com/drive/folders/1ScsRlnzy9Bd_n-xw83SP-0t548v63mPH
    │   └── [train|validation|test]
    │       └── [armchair|cube|greek|vase]
    │           ├── intrinsics.txt
    │           ├── rgb/*.png
    │           └── pose/*.txt
    │
    ├── nerf_llff_data     # Link: https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1
    │   └── [fern|flower|fortress|horns|leaves|orchids|room|trex]
    │
    ├── tanks_and_temples  # Link: https://drive.google.com/file/d/11KRfN91W1AxAW6lOFs4EeYDbeoQZCi87/view?usp=sharing
    │   └── [tat_intermediate_M60|tat_intermediate_Playground|tat_intermediate_Train|tat_training_Truck]
    │       └── [train|test]
    │           ├── intrinsics/*txt
    │           ├── pose/*txt
    │           └── rgb/*jpg
    │
    ├── lf_data            # Link: https://drive.google.com/file/d/1gsjDjkbTh4GAR9fFqlIDZ__qR9NYTURQ/view?usp=sharing
    │   └── [africa|basket|ship|statue|torch]
    │       └── [train|test]
    │           ├── intrinsics/*txt
    │           ├── pose/*txt
    │           └── rgb/*jpg
    │
    ├── 360_v2             # Link: https://jonbarron.info/mipnerf360/
    │   └── [bicycle|bonsai|counter|garden|kitchen|room|stump]
    │       ├── poses_bounds.npy
    │       └── [images_2|images_4]
    │
    ├── nerf_llff_data     # Link: https://drive.google.com/drive/folders/14boI-o5hGO9srnWaaogTU5_ji7wkX2S7
    │   └── [fern|flower|fortress|horns|leaves|orchids|room|trex]
    │       ├── poses_bounds.npy
    │       └── [images_2|images_4]
    │
    └── co3d               # Link: https://github.com/facebookresearch/co3d
        └── [donut|teddybear|umbrella|...]
            ├── frame_annotations.jgz
            ├── set_lists.json
            └── [129_14950_29917|189_20376_35616|...]
                ├── images
                │   └── frame*.jpg
                └── masks
                    └── frame*.png
</details>

## GO

- Training
    ```bash
    $ python run.py --config configs/nerf/lego.py --render_test
    ```
    Use `--i_print` and `--i_weights` to change the log interval.
- Evaluation
    To only evaluate the testset `PSNR`, `SSIM`, and `LPIPS` of the trained `lego` without re-training, run:
    ```bash
    $ python run.py --config configs/nerf/lego.py --render_only --render_test \
                                                  --eval_ssim --eval_lpips_vgg
    ```
    Use `--eval_lpips_alex` to evaluate LPIPS with pre-trained Alex net instead of VGG net.
<!-- - Render video
    ```bash
    $ python run.py --config configs/nerf/lego.py --render_only --render_video
    ```
    Use `--render_video_factor 4` for a fast preview. -->
- Reproduction: all config files to reproduce our results.
    <details>
        <summary> (click to expand) </summary>

        $ ls configs/*
        configs/blendedmvs:
        Character.py  Fountain.py  Jade.py  Statues.py

        configs/nerf:
        chair.py  drums.py  ficus.py  hotdog.py  lego.py  materials.py  mic.py  ship.py

        configs/nsvf:
        Bike.py  Lifestyle.py  Palace.py  Robot.py  Spaceship.py  Steamtrain.py  Toad.py  Wineholder.py

        configs/tankstemple:
        Barn.py  Caterpillar.py  Family.py  Ignatius.py  Truck.py

        configs/deepvoxels:
        armchair.py  cube.py  greek.py  vase.py

        configs/tankstemple_unbounded:
        M60.py  Playground.py  Train.py  Truck.py

        configs/lf:
        africa.py  basket.py  ship.py  statue.py  torch.py

        configs/nerf_unbounded:
        bicycle.py  bonsai.py  counter.py  garden.py  kitchen.py  room.py  stump.py

        configs/llff:
        fern.py  flower.py  fortress.py  horns.py  leaves.py  orchids.py  room.py  trex.py
    </details>
## Rendering With Your Own Code!
`dvgo.render_utils_cuda.sparw` can warp the image from the reference frame to the target frame image, and then use the returned mask to render the occluded areas of the image with your own model.

```python
dvgo.render_utils_cuda.sparw(ref_img, ref_depth, tgt_depth, ref_K, tgt_K, ref_c2w, tgt_c2w, H, W)
```
### PARAMETERS:
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
The code base is origined from the [DirectVoxGO](https://github.com/sunset1995/DirectVoxGO) implementation.
