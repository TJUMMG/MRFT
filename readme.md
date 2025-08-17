#  MRFT: Multiscale Recurrent Fusion Transformer Based Prior Knowledge for Bit-Depth Enhancement
Copyright(c) 2023 Xin Wen
```
If you use this code, please cite the following publication:
Xin Wen, Weizhi Nie, Jing Liu, and Yuting Su, "MRFT: Multiscale Recurrent Fusion Transformer Based Prior Knowledge for Bit-Depth Enhancement", to appear in IEEE Transactions on Circuits and Systems for Video Technology.

```
## Contents

1. [Environment](#1)
2. [Testing](#2)


<h3 id="1">Environment</h3>
Our model is tested through the following environment on Ubuntu:

- Python: 3.8.0
- PyTorch: 1.10.0
- opencv：4.5.5.64

Refer to "MRFT_environment.yml" for the complete environment setup. 

### Testing
We provide four folders "./MRFT_4bit/MRFT_test_4_16", "./MRFT_4bit/MRFT_test_4_8", "./MRFT_6bit/MRFT_test_6_16" and "./MRFT_8bit/MRFT_test_8_16" to realize 4-bit to 16-bit, 4-bit to 8-bit, 6-bit to 16-bit and 8-bit to 16-bit BDE tasks respectively. When testing, prepare the testing dataset, and modify the dataset path and other related content in the code. We provide an image of Sintel dataset (16-bit dataset)  and Kodak dataset (8-bit dataset) respectively for sample testing. You can directly test on the sample image by running-

```
$ python main.py \
--test_only
```
If you want to save the predicted high bit-depth images (--save_results) and high bit-depth ground truths (--save_gt), you can  run-

```
$ python main.py \
--test_only \
--save_results \
--save_gt
```

Note: 

1. We provide recovery results of  sample images in the folder "result" of each models. When testing, the predicted results are saved in the folder "test_stage2" .
2. The files "./metrics/csnr_bits.m" and "./metrics/cal_ssim_bits.m" are used to calculate PSNR and SSIM, respectively.
