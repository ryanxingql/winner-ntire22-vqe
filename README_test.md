# README

NOTE: This document details the inference of our approach on the challenge data for submission. To run our approach on your data, please modify the following scripts.

We provide our data on [Dropbox](https://www.dropbox.com/sh/990dg5sbg0n7cev/AAAabT-HhPwUT6g6nB-z7-7Ma?dl=0):

- Validation set for Track 1
- Validation set for Track 2
- Validation set for Track 3
- Stage 1 model
- Stage 2 model
- FFmpeg

Data structure:

```md
ntire-submission/
|-- stage1/
`-- stage2/
```

If you find this work helpful, please cite our workshop paper.

## 1. Data preparation

```bash
# fixme: assume we put the validation zip at /mnt/nfs1/qunliang/data/submission/
cd /mnt/nfs1/qunliang/data/submission/

unzip validation_track1.zip
unzip validation_track2.zip
unzip validation_track3.zip

# fixme: assume the ffmpeg is located at ./ffmpeg-4.3.1-amd64-static/ffmpeg
for idx in `seq -f '%03g' 1 15`; do mkdir Track1/$idx && ./ffmpeg-4.3.1-amd64-static/ffmpeg -i Track1/$idx.mkv Track1/$idx/f%3d.png; done
for idx in `seq -f '%03g' 1 15`; do mkdir Track2/$idx && ./ffmpeg-4.3.1-amd64-static/ffmpeg -i Track2/$idx.mkv Track2/$idx/f%3d.png; done
for idx in `seq -f '%03g' 1 15`; do mkdir Track3/$idx && ./ffmpeg-4.3.1-amd64-static/ffmpeg -i Track3/$idx.mkv Track3/$idx/f%3d.png; done

cd ntire-submission/stage1/
ln -s /mnt/nfs1/qunliang/data/submission data
```

## 2. Create environment

```bash
# assume we have cuda 10.2
conda create -n mmtest python=3.7 -y && conda activate mmtest
pip install mmcv-full==1.3.14 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.8.0/index.html

cd ntire-submission/ && conda env update -n mmtest --file conda_mmtest.yml

cd ntire-submission/stage1/ && python setup.py develop
```

The following codes are run with the `mmtest` environment.

## 3. Track 1

### 3.1. Generate x8tta lq data

```bash
cd ntire-submission/stage1/toolbox_x8tta/
python generate_x8tta_lq.py -inp_dir ../data/Track1 -out_dir ../data/Track1_x8tta
```

### 3.2. Run stage1 model

```bash
cd ntire-submission/stage1/toolbox_test/

# fixme: assume we have gpus 0, 1, 2, 3

python test.py -gpu 0 -inp_dir '../data/Track1_x8tta/origin' -out_dir '../work_dirs/track1/results_x8tta_rmdup2/origin' -config_path '../configs/restorers/submission/flow5v2r6_ldv_ql_mse1e-6_track1.py' -model_path '../work_dirs/track1/iter_16000.pth' -if_rmdup2 && \
python test.py -gpu 0 -inp_dir '../data/Track1_x8tta/r90' -out_dir '../work_dirs/track1/results_x8tta_rmdup2/r90' -config_path '../configs/restorers/submission/flow5v2r6_ldv_ql_mse1e-6_track1.py' -model_path '../work_dirs/track1/iter_16000.pth' -if_rmdup2

python test.py -gpu 1 -inp_dir '../data/Track1_x8tta/r180' -out_dir '../work_dirs/track1/results_x8tta_rmdup2/r180' -config_path '../configs/restorers/submission/flow5v2r6_ldv_ql_mse1e-6_track1.py' -model_path '../work_dirs/track1/iter_16000.pth' -if_rmdup2 && \
python test.py -gpu 1 -inp_dir '../data/Track1_x8tta/r270' -out_dir '../work_dirs/track1/results_x8tta_rmdup2/r270' -config_path '../configs/restorers/submission/flow5v2r6_ldv_ql_mse1e-6_track1.py' -model_path '../work_dirs/track1/iter_16000.pth' -if_rmdup2

python test.py -gpu 2 -inp_dir '../data/Track1_x8tta/flip' -out_dir '../work_dirs/track1/results_x8tta_rmdup2/flip' -config_path '../configs/restorers/submission/flow5v2r6_ldv_ql_mse1e-6_track1.py' -model_path '../work_dirs/track1/iter_16000.pth' -if_rmdup2 && \
python test.py -gpu 2 -inp_dir '../data/Track1_x8tta/flipr90' -out_dir '../work_dirs/track1/results_x8tta_rmdup2/flipr90' -config_path '../configs/restorers/submission/flow5v2r6_ldv_ql_mse1e-6_track1.py' -model_path '../work_dirs/track1/iter_16000.pth' -if_rmdup2

python test.py -gpu 3 -inp_dir '../data/Track1_x8tta/flipr180' -out_dir '../work_dirs/track1/results_x8tta_rmdup2/flipr180' -config_path '../configs/restorers/submission/flow5v2r6_ldv_ql_mse1e-6_track1.py' -model_path '../work_dirs/track1/iter_16000.pth' -if_rmdup2 && \
python test.py -gpu 3 -inp_dir '../data/Track1_x8tta/flipr270' -out_dir '../work_dirs/track1/results_x8tta_rmdup2/flipr270' -config_path '../configs/restorers/submission/flow5v2r6_ldv_ql_mse1e-6_track1.py' -model_path '../work_dirs/track1/iter_16000.pth' -if_rmdup2
```

### 3.3. Check PSNR vs. LQ and replace videos

```bash
cd ntire-submission/stage1/toolbox_test/

python check_psnr.py -lq_dir '../data/Track1_x8tta/origin' -enh_dir '../work_dirs/track1/results_x8tta_rmdup2/origin' -tab 'track1_model1'
```

Check `ntire-submission/stage1/toolbox_test/log/track1_model1.log`:

1. If this file is empty, go to Section 3.4.
2. If any video appears in this file, run the backup model:

```bash
cd ntire-submission/stage1/toolbox_test/

# fixme: assume we have gpus 0, 1, 2, 3

python test.py -gpu 0 -inp_dir '../data/Track1_x8tta/origin' -out_dir '../work_dirs/track1/results_x8tta_rmdup2_backup/origin' -config_path '../configs/restorers/submission/s1t1_iter_1000.py' -model_path '../work_dirs/track1/iter_1000.pth' -if_rmdup2 && \
python test.py -gpu 0 -inp_dir '../data/Track1_x8tta/r90' -out_dir '../work_dirs/track1/results_x8tta_rmdup2_backup/r90' -config_path '../configs/restorers/submission/s1t1_iter_1000.py' -model_path '../work_dirs/track1/iter_1000.pth' -if_rmdup2

python test.py -gpu 1 -inp_dir '../data/Track1_x8tta/r180' -out_dir '../work_dirs/track1/results_x8tta_rmdup2_backup/r180' -config_path '../configs/restorers/submission/s1t1_iter_1000.py' -model_path '../work_dirs/track1/iter_1000.pth' -if_rmdup2 && \
python test.py -gpu 1 -inp_dir '../data/Track1_x8tta/r270' -out_dir '../work_dirs/track1/results_x8tta_rmdup2_backup/r270' -config_path '../configs/restorers/submission/s1t1_iter_1000.py' -model_path '../work_dirs/track1/iter_1000.pth' -if_rmdup2

python test.py -gpu 2 -inp_dir '../data/Track1_x8tta/flip' -out_dir '../work_dirs/track1/results_x8tta_rmdup2_backup/flip' -config_path '../configs/restorers/submission/s1t1_iter_1000.py' -model_path '../work_dirs/track1/iter_1000.pth' -if_rmdup2 && \
python test.py -gpu 2 -inp_dir '../data/Track1_x8tta/flipr90' -out_dir '../work_dirs/track1/results_x8tta_rmdup2_backup/flipr90' -config_path '../configs/restorers/submission/s1t1_iter_1000.py' -model_path '../work_dirs/track1/iter_1000.pth' -if_rmdup2

python test.py -gpu 3 -inp_dir '../data/Track1_x8tta/flipr180' -out_dir '../work_dirs/track1/results_x8tta_rmdup2_backup/flipr180' -config_path '../configs/restorers/submission/s1t1_iter_1000.py' -model_path '../work_dirs/track1/iter_1000.pth' -if_rmdup2 && \
python test.py -gpu 3 -inp_dir '../data/Track1_x8tta/flipr270' -out_dir '../work_dirs/track1/results_x8tta_rmdup2_backup/flipr270' -config_path '../configs/restorers/submission/s1t1_iter_1000.py' -model_path '../work_dirs/track1/iter_1000.pth' -if_rmdup2
```

Then replace the videos at list `ntire-submission/stage1/toolbox_test/log/track1_model1.log` and in `ntire-submission/stage1/work_dirs/track1/results_x8tta_rmdup2/` with those in `ntire-submission/stage1/work_dirs/track1/results_x8tta_rmdup2_backup`.

### 3.4. Generate image symbolic link dir for stage2

```bash
cd ntire-submission/stage2/toolbox_data/
```

To test full set:

```bash
python rename_and_ln.py -inp_dir '../../stage1/work_dirs/track1/results_x8tta_rmdup2' -out_dir '../work_dirs/track1/input_x8tta_rmdup2'
```

Or test online validation set (f010, f020, ...):

```bash
python rename_and_ln.py -inp_dir '../../stage1/work_dirs/track1/results_x8tta_rmdup2' -out_dir '../work_dirs/track1/input_x8tta_rmdup2_online10' -online
```

### 3.5. Run stage2 model

To test full set:

```bash
# fixme: assume we have gpus 0, 1, 2, 3
cd ntire-submission/stage2/

python main_test_swinir.py --task color_dn --noise 50 --model_path 'work_dirs/track1/9000_G.pth' --folder_lq 'work_dirs/track1/input_x8tta_rmdup2/origin' --folder_output 'work_dirs/track1/results_x8tta_rmdup2_npy/origin' --gpus 0 --save_npy True && \
python main_test_swinir.py --task color_dn --noise 50 --model_path 'work_dirs/track1/9000_G.pth' --folder_lq 'work_dirs/track1/input_x8tta_rmdup2/r90' --folder_output 'work_dirs/track1/results_x8tta_rmdup2_npy/r90' --gpus 0 --save_npy True

python main_test_swinir.py --task color_dn --noise 50 --model_path 'work_dirs/track1/9000_G.pth' --folder_lq 'work_dirs/track1/input_x8tta_rmdup2/r180' --folder_output 'work_dirs/track1/results_x8tta_rmdup2_npy/r180' --gpus 1 --save_npy True && \
python main_test_swinir.py --task color_dn --noise 50 --model_path 'work_dirs/track1/9000_G.pth' --folder_lq 'work_dirs/track1/input_x8tta_rmdup2/r270' --folder_output 'work_dirs/track1/results_x8tta_rmdup2_npy/r270' --gpus 1 --save_npy True

python main_test_swinir.py --task color_dn --noise 50 --model_path 'work_dirs/track1/9000_G.pth' --folder_lq 'work_dirs/track1/input_x8tta_rmdup2/flip' --folder_output 'work_dirs/track1/results_x8tta_rmdup2_npy/flip' --gpus 2 --save_npy True && \
python main_test_swinir.py --task color_dn --noise 50 --model_path 'work_dirs/track1/9000_G.pth' --folder_lq 'work_dirs/track1/input_x8tta_rmdup2/flipr90' --folder_output 'work_dirs/track1/results_x8tta_rmdup2_npy/flipr90' --gpus 2 --save_npy True

python main_test_swinir.py --task color_dn --noise 50 --model_path 'work_dirs/track1/9000_G.pth' --folder_lq 'work_dirs/track1/input_x8tta_rmdup2/flipr180' --folder_output 'work_dirs/track1/results_x8tta_rmdup2_npy/flipr180' --gpus 3 --save_npy True && \
python main_test_swinir.py --task color_dn --noise 50 --model_path 'work_dirs/track1/9000_G.pth' --folder_lq 'work_dirs/track1/input_x8tta_rmdup2/flipr270' --folder_output 'work_dirs/track1/results_x8tta_rmdup2_npy/flipr270' --gpus 3 --save_npy True
```

Or test online validation set (f010, f020, ...):

```bash
# fixme: assume we have gpus 0, 1, 2, 3
cd ntire-submission/stage2/

python main_test_swinir.py --task color_dn --noise 50 --model_path 'work_dirs/track1/9000_G.pth' --folder_lq 'work_dirs/track1/input_x8tta_rmdup2_online10/origin' --folder_output 'work_dirs/track1/results_x8tta_rmdup2_npy_online10/origin' --gpus 0 --save_npy True && \
python main_test_swinir.py --task color_dn --noise 50 --model_path 'work_dirs/track1/9000_G.pth' --folder_lq 'work_dirs/track1/input_x8tta_rmdup2_online10/r90' --folder_output 'work_dirs/track1/results_x8tta_rmdup2_npy_online10/r90' --gpus 0 --save_npy True

python main_test_swinir.py --task color_dn --noise 50 --model_path 'work_dirs/track1/9000_G.pth' --folder_lq 'work_dirs/track1/input_x8tta_rmdup2_online10/r180' --folder_output 'work_dirs/track1/results_x8tta_rmdup2_npy_online10/r180' --gpus 1 --save_npy True && \
python main_test_swinir.py --task color_dn --noise 50 --model_path 'work_dirs/track1/9000_G.pth' --folder_lq 'work_dirs/track1/input_x8tta_rmdup2_online10/r270' --folder_output 'work_dirs/track1/results_x8tta_rmdup2_npy_online10/r270' --gpus 1 --save_npy True

python main_test_swinir.py --task color_dn --noise 50 --model_path 'work_dirs/track1/9000_G.pth' --folder_lq 'work_dirs/track1/input_x8tta_rmdup2_online10/flip' --folder_output 'work_dirs/track1/results_x8tta_rmdup2_npy_online10/flip' --gpus 2 --save_npy True && \
python main_test_swinir.py --task color_dn --noise 50 --model_path 'work_dirs/track1/9000_G.pth' --folder_lq 'work_dirs/track1/input_x8tta_rmdup2_online10/flipr90' --folder_output 'work_dirs/track1/results_x8tta_rmdup2_npy_online10/flipr90' --gpus 2 --save_npy True

python main_test_swinir.py --task color_dn --noise 50 --model_path 'work_dirs/track1/9000_G.pth' --folder_lq 'work_dirs/track1/input_x8tta_rmdup2_online10/flipr180' --folder_output 'work_dirs/track1/results_x8tta_rmdup2_npy_online10/flipr180' --gpus 3 --save_npy True && \
python main_test_swinir.py --task color_dn --noise 50 --model_path 'work_dirs/track1/9000_G.pth' --folder_lq 'work_dirs/track1/input_x8tta_rmdup2_online10/flipr270' --folder_output 'work_dirs/track1/results_x8tta_rmdup2_npy_online10/flipr270' --gpus 3 --save_npy True
```

### 3.6. Merge x8tta npy files into images in one dir

To test full set:

```bash
cd ntire-submission/stage2/toolbox_data/
python merge_npy.py -inp_dir '../work_dirs/track1/results_x8tta_rmdup2_npy' -out_dir '../work_dirs/track1/results_x8tta_rmdup2_npy/merge'
```

Or test online validation set (f010, f020, ...):

```bash
cd ntire-submission/stage2/toolbox_data/
python merge_npy.py -inp_dir '../work_dirs/track1/results_x8tta_rmdup2_npy_online10' -out_dir '../work_dirs/track1/results_x8tta_rmdup2_npy_online10/merge'
```

### 3.7. Collect images to each video dir

To test full set:

```bash
cd ntire-submission/stage2/toolbox_data/
python collect_and_submit.py -inp_dir '../work_dirs/track1/results_x8tta_rmdup2_npy/merge' -out_dir '../work_dirs/track1/results_final'
```

The final results are stored at: `ntire-submission/stage2/work_dirs/track1/results_final`

Or test online validation set (f010, f020, ...):

```bash
cd ntire-submission/stage2/toolbox_data/
python collect_and_submit.py -inp_dir '../work_dirs/track1/results_x8tta_rmdup2_npy_online10/merge' -out_dir '../work_dirs/track1/results_final_online10'
```

The final results are stored at: `ntire-submission/stage2/work_dirs/track1/results_final_online10`

## 4. Track 2

### 4.1. Generate x8tta lq data

```bash
cd ntire-submission/stage1/toolbox_x8tta/
python generate_x8tta_lq.py -inp_dir ../data/Track2 -out_dir ../data/Track2_x8tta
```

### 4.2. Run stage1 model

```bash
cd ntire-submission/stage1/toolbox_test/

# fixme: assume we have gpus 0, 1, 2, 3

python test.py -gpu 0 -inp_dir '../data/Track2_x8tta/origin' -out_dir '../work_dirs/track2/results_x8tta/origin' -config_path '../configs/restorers/submission/flow5v2r5_track2_newdata_mse_ldv.py' -model_path '../work_dirs/track2/iter_3000.pth' && \
python test.py -gpu 0 -inp_dir '../data/Track2_x8tta/r90' -out_dir '../work_dirs/track2/results_x8tta/r90' -config_path '../configs/restorers/submission/flow5v2r5_track2_newdata_mse_ldv.py' -model_path '../work_dirs/track2/iter_3000.pth'

python test.py -gpu 1 -inp_dir '../data/Track2_x8tta/r180' -out_dir '../work_dirs/track2/results_x8tta/r180' -config_path '../configs/restorers/submission/flow5v2r5_track2_newdata_mse_ldv.py' -model_path '../work_dirs/track2/iter_3000.pth' && \
python test.py -gpu 1 -inp_dir '../data/Track2_x8tta/r270' -out_dir '../work_dirs/track2/results_x8tta/r270' -config_path '../configs/restorers/submission/flow5v2r5_track2_newdata_mse_ldv.py' -model_path '../work_dirs/track2/iter_3000.pth'

python test.py -gpu 2 -inp_dir '../data/Track2_x8tta/flip' -out_dir '../work_dirs/track2/results_x8tta/flip' -config_path '../configs/restorers/submission/flow5v2r5_track2_newdata_mse_ldv.py' -model_path '../work_dirs/track2/iter_3000.pth' && \
python test.py -gpu 2 -inp_dir '../data/Track2_x8tta/flipr90' -out_dir '../work_dirs/track2/results_x8tta/flipr90' -config_path '../configs/restorers/submission/flow5v2r5_track2_newdata_mse_ldv.py' -model_path '../work_dirs/track2/iter_3000.pth'

python test.py -gpu 3 -inp_dir '../data/Track2_x8tta/flipr180' -out_dir '../work_dirs/track2/results_x8tta/flipr180' -config_path '../configs/restorers/submission/flow5v2r5_track2_newdata_mse_ldv.py' -model_path '../work_dirs/track2/iter_3000.pth' && \
python test.py -gpu 3 -inp_dir '../data/Track2_x8tta/flipr270' -out_dir '../work_dirs/track2/results_x8tta/flipr270' -config_path '../configs/restorers/submission/flow5v2r5_track2_newdata_mse_ldv.py' -model_path '../work_dirs/track2/iter_3000.pth'
```

### 4.3. Generate image symbolic link dir for stage2

```bash
cd ntire-submission/stage2/toolbox_data/
```

To test full set:

```bash
python rename_and_ln.py -inp_dir '../../stage1/work_dirs/track2/results_x8tta' -out_dir '../work_dirs/track2/input_x8tta'
```

Or test online validation set (f010, f020, ...):

```bash
python rename_and_ln.py -inp_dir '../../stage1/work_dirs/track2/results_x8tta' -out_dir '../work_dirs/track2/input_x8tta_online10' -online
```

### 4.4. Run stage2 model

To test full set:

```bash
# fixme: assume we have gpus 0, 1, 2, 3
cd ntire-submission/stage2/

python main_test_swinir.py --task color_dn --noise 50 --model_path 'work_dirs/track2/15000_G.pth' --folder_lq 'work_dirs/track2/input_x8tta/origin' --folder_output 'work_dirs/track2/results_x8tta_npy/origin' --gpus 0 --save_npy True && \
python main_test_swinir.py --task color_dn --noise 50 --model_path 'work_dirs/track2/15000_G.pth' --folder_lq 'work_dirs/track2/input_x8tta/r90' --folder_output 'work_dirs/track2/results_x8tta_npy/r90' --gpus 0 --save_npy True

python main_test_swinir.py --task color_dn --noise 50 --model_path 'work_dirs/track2/15000_G.pth' --folder_lq 'work_dirs/track2/input_x8tta/r180' --folder_output 'work_dirs/track2/results_x8tta_npy/r180' --gpus 1 --save_npy True && \
python main_test_swinir.py --task color_dn --noise 50 --model_path 'work_dirs/track2/15000_G.pth' --folder_lq 'work_dirs/track2/input_x8tta/r270' --folder_output 'work_dirs/track2/results_x8tta_npy/r270' --gpus 1 --save_npy True

python main_test_swinir.py --task color_dn --noise 50 --model_path 'work_dirs/track2/15000_G.pth' --folder_lq 'work_dirs/track2/input_x8tta/flip' --folder_output 'work_dirs/track2/results_x8tta_npy/flip' --gpus 2 --save_npy True && \
python main_test_swinir.py --task color_dn --noise 50 --model_path 'work_dirs/track2/15000_G.pth' --folder_lq 'work_dirs/track2/input_x8tta/flipr90' --folder_output 'work_dirs/track2/results_x8tta_npy/flipr90' --gpus 2 --save_npy True

python main_test_swinir.py --task color_dn --noise 50 --model_path 'work_dirs/track2/15000_G.pth' --folder_lq 'work_dirs/track2/input_x8tta/flipr180' --folder_output 'work_dirs/track2/results_x8tta_npy/flipr180' --gpus 3 --save_npy True && \
python main_test_swinir.py --task color_dn --noise 50 --model_path 'work_dirs/track2/15000_G.pth' --folder_lq 'work_dirs/track2/input_x8tta/flipr270' --folder_output 'work_dirs/track2/results_x8tta_npy/flipr270' --gpus 3 --save_npy True
```

Or test online validation set (f010, f020, ...):

```bash
# fixme: assume we have gpus 0, 1, 2, 3
cd ntire-submission/stage2/

python main_test_swinir.py --task color_dn --noise 50 --model_path 'work_dirs/track2/15000_G.pth' --folder_lq 'work_dirs/track2/input_x8tta_online10/origin' --folder_output 'work_dirs/track2/results_x8tta_npy_online10/origin' --gpus 0 --save_npy True && \
python main_test_swinir.py --task color_dn --noise 50 --model_path 'work_dirs/track2/15000_G.pth' --folder_lq 'work_dirs/track2/input_x8tta_online10/r90' --folder_output 'work_dirs/track2/results_x8tta_npy_online10/r90' --gpus 0 --save_npy True

python main_test_swinir.py --task color_dn --noise 50 --model_path 'work_dirs/track2/15000_G.pth' --folder_lq 'work_dirs/track2/input_x8tta_online10/r180' --folder_output 'work_dirs/track2/results_x8tta_npy_online10/r180' --gpus 1 --save_npy True && \
python main_test_swinir.py --task color_dn --noise 50 --model_path 'work_dirs/track2/15000_G.pth' --folder_lq 'work_dirs/track2/input_x8tta_online10/r270' --folder_output 'work_dirs/track2/results_x8tta_npy_online10/r270' --gpus 1 --save_npy True

python main_test_swinir.py --task color_dn --noise 50 --model_path 'work_dirs/track2/15000_G.pth' --folder_lq 'work_dirs/track2/input_x8tta_online10/flip' --folder_output 'work_dirs/track2/results_x8tta_npy_online10/flip' --gpus 2 --save_npy True && \
python main_test_swinir.py --task color_dn --noise 50 --model_path 'work_dirs/track2/15000_G.pth' --folder_lq 'work_dirs/track2/input_x8tta_online10/flipr90' --folder_output 'work_dirs/track2/results_x8tta_npy_online10/flipr90' --gpus 2 --save_npy True

python main_test_swinir.py --task color_dn --noise 50 --model_path 'work_dirs/track2/15000_G.pth' --folder_lq 'work_dirs/track2/input_x8tta_online10/flipr180' --folder_output 'work_dirs/track2/results_x8tta_npy_online10/flipr180' --gpus 3 --save_npy True && \
python main_test_swinir.py --task color_dn --noise 50 --model_path 'work_dirs/track2/15000_G.pth' --folder_lq 'work_dirs/track2/input_x8tta_online10/flipr270' --folder_output 'work_dirs/track2/results_x8tta_npy_online10/flipr270' --gpus 3 --save_npy True
```

### 4.5. Merge x8tta npy files into images in one dir

To test full set:

```bash
cd ntire-submission/stage2/toolbox_data/
python merge_npy.py -inp_dir '../work_dirs/track2/results_x8tta_npy' -out_dir '../work_dirs/track2/results_x8tta_npy/merge'
```

Or test online validation set (f010, f020, ...):

```bash
cd ntire-submission/stage2/toolbox_data/
python merge_npy.py -inp_dir '../work_dirs/track2/results_x8tta_npy_online10' -out_dir '../work_dirs/track2/results_x8tta_npy_online10/merge'
```

### 4.6. Collect images to each video dir

To test full set:

```bash
cd ntire-submission/stage2/toolbox_data/
python collect_and_submit.py -inp_dir '../work_dirs/track2/results_x8tta_npy/merge' -out_dir '../work_dirs/track2/results_final'
```

The final results are stored at: `ntire-submission/stage2/work_dirs/track2/results_final`

Or test online validation set (f010, f020, ...):

```bash
cd ntire-submission/stage2/toolbox_data/
python collect_and_submit.py -inp_dir '../work_dirs/track2/results_x8tta_npy_online10/merge' -out_dir '../work_dirs/track2/results_final_online10'
```

The final results are stored at: `ntire-submission/stage2/work_dirs/track2/results_final_online10`

## 5. Track 3

### 5.1. Generate x8tta lq data

```bash
cd ntire-submission/stage1/toolbox_x8tta/
python generate_x8tta_lq.py -inp_dir ../data/Track3 -out_dir ../data/Track3_x8tta
```

### 5.2. Run stage1 modelx2 (model ensemble here)

Model 1:

```bash
cd ntire-submission/stage1/toolbox_test/

# fixme: assume we have gpus 0, 1, 2, 3

python test.py -gpu 0 -inp_dir '../data/Track3_x8tta/origin' -out_dir '../work_dirs/track3/results_x8tta_rmdup2_model1/origin' -config_path '../configs/restorers/submission/c128n25rec3_track3mse.py' -model_path '../work_dirs/track3/iter_11000.pth' -float -out_npy_dir '../work_dirs/track3/results_x8tta_rmdup2_npy_model1/origin' -if_rmdup2 && \
python test.py -gpu 0 -inp_dir '../data/Track3_x8tta/r90' -out_dir '../work_dirs/track3/results_x8tta_rmdup2_model1/r90' -config_path '../configs/restorers/submission/c128n25rec3_track3mse.py' -model_path '../work_dirs/track3/iter_11000.pth' -float -out_npy_dir '../work_dirs/track3/results_x8tta_rmdup2_npy_model1/r90' -if_rmdup2

python test.py -gpu 1 -inp_dir '../data/Track3_x8tta/r180' -out_dir '../work_dirs/track3/results_x8tta_rmdup2_model1/r180' -config_path '../configs/restorers/submission/c128n25rec3_track3mse.py' -model_path '../work_dirs/track3/iter_11000.pth' -float -out_npy_dir '../work_dirs/track3/results_x8tta_rmdup2_npy_model1/r180' -if_rmdup2 && \
python test.py -gpu 1 -inp_dir '../data/Track3_x8tta/r270' -out_dir '../work_dirs/track3/results_x8tta_rmdup2_model1/r270' -config_path '../configs/restorers/submission/c128n25rec3_track3mse.py' -model_path '../work_dirs/track3/iter_11000.pth' -float -out_npy_dir '../work_dirs/track3/results_x8tta_rmdup2_npy_model1/r270' -if_rmdup2

python test.py -gpu 2 -inp_dir '../data/Track3_x8tta/flip' -out_dir '../work_dirs/track3/results_x8tta_rmdup2_model1/flip' -config_path '../configs/restorers/submission/c128n25rec3_track3mse.py' -model_path '../work_dirs/track3/iter_11000.pth' -float -out_npy_dir '../work_dirs/track3/results_x8tta_rmdup2_npy_model1/flip' -if_rmdup2 && \
python test.py -gpu 2 -inp_dir '../data/Track3_x8tta/flipr90' -out_dir '../work_dirs/track3/results_x8tta_rmdup2_model1/flipr90' -config_path '../configs/restorers/submission/c128n25rec3_track3mse.py' -model_path '../work_dirs/track3/iter_11000.pth' -float -out_npy_dir '../work_dirs/track3/results_x8tta_rmdup2_npy_model1/flipr90' -if_rmdup2

python test.py -gpu 3 -inp_dir '../data/Track3_x8tta/flipr180' -out_dir '../work_dirs/track3/results_x8tta_rmdup2_model1/flipr180' -config_path '../configs/restorers/submission/c128n25rec3_track3mse.py' -model_path '../work_dirs/track3/iter_11000.pth' -float -out_npy_dir '../work_dirs/track3/results_x8tta_rmdup2_npy_model1/flipr180' -if_rmdup2 && \
python test.py -gpu 3 -inp_dir '../data/Track3_x8tta/flipr270' -out_dir '../work_dirs/track3/results_x8tta_rmdup2_model1/flipr270' -config_path '../configs/restorers/submission/c128n25rec3_track3mse.py' -model_path '../work_dirs/track3/iter_11000.pth' -float -out_npy_dir '../work_dirs/track3/results_x8tta_rmdup2_npy_model1/flipr270' -if_rmdup2
```

Merge:

```bash
python merge_npy.py -inp_dir '../work_dirs/track3/results_x8tta_rmdup2_npy_model1' -out_dir '../work_dirs/track3/results_x8tta_rmdup2_npy_model1_merge'
```

Model 2:

```bash
cd ntire-submission/stage1/toolbox_test/

# fixme: assume we have gpus 0, 1, 2, 3

python test.py -gpu 0 -inp_dir '../data/Track3_x8tta/origin' -out_dir '../work_dirs/track3/results_x8tta_rmdup2_model2/origin' -config_path '../configs/restorers/submission/flow5v2r6_track3mse.py' -model_path '../work_dirs/track3/iter_8000.pth' -float -out_npy_dir '../work_dirs/track3/results_x8tta_rmdup2_npy_model2/origin' -if_rmdup2 && \
python test.py -gpu 0 -inp_dir '../data/Track3_x8tta/r90' -out_dir '../work_dirs/track3/results_x8tta_rmdup2_model2/r90' -config_path '../configs/restorers/submission/flow5v2r6_track3mse.py' -model_path '../work_dirs/track3/iter_8000.pth' -float -out_npy_dir '../work_dirs/track3/results_x8tta_rmdup2_npy_model2/r90' -if_rmdup2

python test.py -gpu 1 -inp_dir '../data/Track3_x8tta/r180' -out_dir '../work_dirs/track3/results_x8tta_rmdup2_model2/r180' -config_path '../configs/restorers/submission/flow5v2r6_track3mse.py' -model_path '../work_dirs/track3/iter_8000.pth' -float -out_npy_dir '../work_dirs/track3/results_x8tta_rmdup2_npy_model2/r180' -if_rmdup2 && \
python test.py -gpu 1 -inp_dir '../data/Track3_x8tta/r270' -out_dir '../work_dirs/track3/results_x8tta_rmdup2_model2/r270' -config_path '../configs/restorers/submission/flow5v2r6_track3mse.py' -model_path '../work_dirs/track3/iter_8000.pth' -float -out_npy_dir '../work_dirs/track3/results_x8tta_rmdup2_npy_model2/r270' -if_rmdup2

python test.py -gpu 2 -inp_dir '../data/Track3_x8tta/flip' -out_dir '../work_dirs/track3/results_x8tta_rmdup2_model2/flip' -config_path '../configs/restorers/submission/flow5v2r6_track3mse.py' -model_path '../work_dirs/track3/iter_8000.pth' -float -out_npy_dir '../work_dirs/track3/results_x8tta_rmdup2_npy_model2/flip' -if_rmdup2 && \
python test.py -gpu 2 -inp_dir '../data/Track3_x8tta/flipr90' -out_dir '../work_dirs/track3/results_x8tta_rmdup2_model2/flipr90' -config_path '../configs/restorers/submission/flow5v2r6_track3mse.py' -model_path '../work_dirs/track3/iter_8000.pth' -float -out_npy_dir '../work_dirs/track3/results_x8tta_rmdup2_npy_model2/flipr90' -if_rmdup2

python test.py -gpu 3 -inp_dir '../data/Track3_x8tta/flipr180' -out_dir '../work_dirs/track3/results_x8tta_rmdup2_model2/flipr180' -config_path '../configs/restorers/submission/flow5v2r6_track3mse.py' -model_path '../work_dirs/track3/iter_8000.pth' -float -out_npy_dir '../work_dirs/track3/results_x8tta_rmdup2_npy_model2/flipr180' -if_rmdup2 && \
python test.py -gpu 3 -inp_dir '../data/Track3_x8tta/flipr270' -out_dir '../work_dirs/track3/results_x8tta_rmdup2_model2/flipr270' -config_path '../configs/restorers/submission/flow5v2r6_track3mse.py' -model_path '../work_dirs/track3/iter_8000.pth' -float -out_npy_dir '../work_dirs/track3/results_x8tta_rmdup2_npy_model2/flipr270' -if_rmdup2
```

Merge:

```bash
python merge_npy.py -inp_dir '../work_dirs/track3/results_x8tta_rmdup2_npy_model2' -out_dir '../work_dirs/track3/results_x8tta_rmdup2_npy_model2_merge'
```

### 5.3. Model ensemble

```bash
cd ntire-submission/stage1/toolbox_test/

python check_psnr.py -lq_dir '../data/Track3_x8tta/origin' -enh_dir '../work_dirs/track3/results_x8tta_rmdup2_npy_model2_merge/png' -tab 'track3_model1'
python check_psnr.py -lq_dir '../data/Track3_x8tta/origin' -enh_dir '../work_dirs/track3/results_x8tta_rmdup2_npy_model2_merge/png' -tab 'track3_model2'

python model_ensemble.py --model1_result_dir '../work_dirs/track3/results_x8tta_rmdup2_npy_model1_merge/npy' --model1_png_dir '../work_dirs/track3/results_x8tta_rmdup2_npy_model1_merge/png' --model1_lst_file 'log/track3_model1.log' --model2_result_dir '../work_dirs/track3/results_x8tta_rmdup2_npy_model2_merge/npy' --model2_png_dir '../work_dirs/track3/results_x8tta_rmdup2_npy_model2_merge/png' --model2_lst_file 'log/track3_model2.log' --save_dir '../work_dirs/track3/results_x8tta_rmdup2_model_ensemble'
```

### 5.4. Generate image symbolic link dir for stage2

```bash
cd ntire-submission/stage2/toolbox_data/
```

To test full set:

```bash
python rename_and_ln_track3.py -inp_dir '../../stage1/work_dirs/track3/results_x8tta_rmdup2_model_ensemble' -out_dir '../work_dirs/track3/input_x8tta_rmdup2'
```

Or test online validation set (f010, f020, ...):

```bash
python rename_and_ln_track3.py -inp_dir '../../stage1/work_dirs/track3/results_x8tta_rmdup2_model_ensemble' -out_dir '../work_dirs/track3/input_x8tta_rmdup2_online10' -online
```

### 5.5. Run stage2 model

To test full set:

```bash
cd ntire-submission/stage2/

python main_test_swinir.py --task color_dn --noise 50 --model_path 'work_dirs/track3/15000_G.pth' --folder_lq 'work_dirs/track3/input_x8tta_rmdup2' --folder_output 'work_dirs/track3/results_x8tta_rmdup2_npy' --gpus 0 --save_npy True
```

Or test online validation set (f010, f020, ...):

```bash
# fixme: assume we have gpus 0, 1, 2, 3
cd ntire-submission/stage2/

python main_test_swinir.py --task color_dn --noise 50 --model_path 'work_dirs/track3/15000_G.pth' --folder_lq 'work_dirs/track3/input_x8tta_rmdup2_online10' --folder_output 'work_dirs/track3/results_x8tta_rmdup2_npy_online10' --gpus 0 --save_npy True
```

### 5.6. Convert npy files into images in one dir

To test full set:

```bash
cd ntire-submission/stage2/toolbox_data/
python merge_npy_track3.py -inp_dir '../work_dirs/track3/results_x8tta_rmdup2_npy' -out_dir '../work_dirs/track3/results_x8tta_rmdup2_npy/merge'
```

Or test online validation set (f010, f020, ...):

```bash
cd ntire-submission/stage2/toolbox_data/
python merge_npy_track3.py -inp_dir '../work_dirs/track3/results_x8tta_rmdup2_npy_online10' -out_dir '../work_dirs/track3/results_x8tta_rmdup2_npy_online10/merge'
```

### 5.7. Collect images to each video dir

To test full set:

```bash
cd ntire-submission/stage2/toolbox_data/
python collect_and_submit.py -inp_dir '../work_dirs/track3/results_x8tta_rmdup2_npy/merge' -out_dir '../work_dirs/track3/results_final'
```

The final results are stored at: `ntire-submission/stage2/work_dirs/track3/results_final`

Or test online validation set (f010, f020, ...):

```bash
cd ntire-submission/stage2/toolbox_data/
python collect_and_submit.py -inp_dir '../work_dirs/track3/results_x8tta_rmdup2_npy_online10/merge' -out_dir '../work_dirs/track3/results_final_online10'
```

The final results are stored at: `ntire-submission/stage2/work_dirs/track3/results_final_online10`
