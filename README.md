# Mip-Nerf 360-pytorch: An unofficial port of the JAX-based Mip-NeRF 360 code release to PyTorch

You can find the original jax version released by Google at [https://github.com/google-research/multinerf](https://github.com/google-research/multinerf)

*This is not an officially supported Google product.*

This repository contains the code release for the CVPR2022 Mip-NeRF 360 paper:[Mip-NeRF-360](https://jonbarron.info/mipnerf360/)
This codebase was adapted from the [multinerf](https://github.com/google/multinerf) code release that combines the mip-NeRF-360, Raw-NeRF, Ref-NeRF papers from CVPR 2022.
to [PyTorch](https://github.com/pytorch/pytorch) the results might be different.

## Setup

```shell
# Clone the repo.
git clone https://github.com/ingra14m/mipnerf360-pytorch
cd mipnerf360-pytorch

# Make a conda environment.
conda create --name mipnerf360 python=3.9
conda activate mipnerf360
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116

# Install requirements.
pip install -r requirements.txt

# Manually install rmbrualla's `pycolmap` (don't use pip's! It's different).
git clone https://github.com/rmbrualla/pycolmap.git ./internal/pycolmap
```

## Running

You'll need to change the paths to point to wherever the datasets are located. [Gin](https://github.com/google/gin-config) configuration files for our model and some ablations can be found in `configs/`.

### OOM errors

You may need to reduce the batch size (`Config.batch_size`) to avoid out of memory
errors. If you do this, but want to preserve quality, be sure to increase the number
of training iterations and decrease the learning rate by whatever scale factor you
decrease batch size by.

## Using your own data

Summary: first, calculate poses. Second, train Ref-NeRF. Third, render a result video from the trained NeRF model.

1. Calculating poses (using COLMAP):
```shell
DATA_DIR=my_dataset_dir
bash scripts/local_colmap_and_resize.sh ${DATA_DIR}
EXP_NAME=bicycle
```
2. Training Mip-NeRF-360:
```shell
python -m train \
  --gin_configs=configs/360.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
  --gin_bindings="Config.checkpoint_dir = 'logs/${EXP_NAME}/checkpoints'" \
  --gin_bindings="Config.saveimage_dir = 'logs/${EXP_NAME}/images'"
  --logtostderr
```
3. Rendering Mip-NeRF-360(Not test yet):
```
python -m render \
  --gin_configs=configs/360.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
  --gin_bindings="Config.checkpoint_dir = 'logs/${EXP_NAME}/checkpoints'" \
  --gin_bindings="Config.render_dir = 'logs/${EXP_NAME}/render'" \
  --gin_bindings="Config.render_path = True" \
  --gin_bindings="Config.render_path_frames = 480" \
  --gin_bindings="Config.render_video_fps = 60" \
  --logtostderr
```
Your output video should now exist in the directory `logs/${EXP_NAME}/render/`.

See below for more detailed instructions on either using COLMAP to calculate poses or writing your own dataset loader (if you already have pose data from another source, like SLAM or RealityCapture).

### Running COLMAP to get camera poses

In order to run Ref-NeRF on your own captured images of a scene, you must first run [COLMAP](https://colmap.github.io/install.html) to calculate camera poses. You can do this using our provided script `scripts/local_colmap_and_resize.sh`. Just make a directory `my_dataset_dir/` and copy your input images into a folder `my_dataset_dir/images/`, then run:
```shell
bash scripts/local_colmap_and_resize.sh my_dataset_dir
```
This will run COLMAP and create 2x, 4x, and 8x downsampled versions of your images. These lower resolution images can be used in NeRF by setting, e.g., the `Config.factor = 4` gin flag.

By default, `local_colmap_and_resize.sh` uses the OPENCV camera model, which is a perspective pinhole camera with k1, k2 radial and t1, t2 tangential distortion coefficients. To switch to another COLMAP camera model, for example OPENCV_FISHEYE, you can run
```shell
bash scripts/local_colmap_and_resize.sh my_dataset_dir OPENCV_FISHEYE
```

If you have a very large capture of more than around 500 images, we recommend switching from the exhaustive matcher to the vocabulary tree matcher in COLMAP (see the script for a commented-out example).

Our script is simply a thin wrapper for COLMAP--if you have run COLMAP yourself, all you need to do to load your scene in NeRF is ensure it has the following format:
```shell
my_dataset_dir/images/    <--- all input images
my_dataset_dir/sparse/0/  <--- COLMAP sparse reconstruction files (cameras, images, points)
```

### Details of the inner workings of Dataset

The public interface mimics the behavior of a standard machine learning pipeline
dataset provider that can provide infinite batches of data to the
training/testing pipelines without exposing any details of how the batches are
loaded/created or how this is parallelized. Therefore, the initializer runs all
setup, including data loading from disk using `_load_renderings`, and begins
the thread using its parent start() method. After the initializer returns, the
caller can request batches of data straight away.

The internal `self._queue` is initialized as `queue.Queue(3)`, so the infinite
loop in `run()` will block on the call `self._queue.put(self._next_fn())` once
there are 3 elements. The main thread training job runs in a loop that pops 1
element at a time off the front of the queue. The Dataset thread's `run()` loop
will populate the queue with 3 elements, then wait until a batch has been
removed and push one more onto the end.

This repeats indefinitely until the main thread's training loop completes
(typically hundreds of thousands of iterations), then the main thread will exit
and the Dataset thread will automatically be killed since it is a daemon.

## References

```shell
@article{barron2022mipnerf360,
    title={Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields},
    author={Jonathan T. Barron and Ben Mildenhall and 
            Dor Verbin and Pratul P. Srinivasan and Peter Hedman},
    journal={CVPR},
    year={2022}
}
```
```shell
@misc{multinerf2022,
      title={{MultiNeRF}: {A} {Code} {Release} for {Mip-NeRF} 360, {Ref-NeRF}, and {RawNeRF}},
      author={Ben Mildenhall and Dor Verbin and Pratul P. Srinivasan and Peter Hedman and Ricardo Martin-Brualla and Jonathan T. Barron},
      year={2022},
      url={https://github.com/google-research/multinerf},
}
```

```shell
@misc{refnerf-pytorch,
      title={refnerf-pytorch: A port of Ref-NeRF from jax to pytorch},
      author={Georgios Kouros},
      year={2022},
      url={https://github.com/google-research/refnerf-pytorch},
}
```