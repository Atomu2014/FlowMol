1. convert sdf to pkl

bpa_sdf_to_pkl.py

2. split bpa.pkl into train/val/test_data.pkl

bpa_split_pkl.py

3. process data

python process_geom.py data/bpa_raw/train_data.pkl --config=configs/bpa_finetune.yml
python process_geom.py data/bpa_raw/val_data.pkl --config=configs/bpa_finetune.yml
python process_geom.py data/bpa_raw/test_data.pkl --config=configs/bpa_finetune.yml

4. finetune with 1 gpu

CUDA_VISIBLE_DEVICES=0 python train.py --config configs/bpa_finetune.yml
