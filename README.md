# SwinCheX

This repo is the official implementation of ["SwinCheX: Multi-label classification on chest X-ray images with transformer"](). (Link will be added soon)


## Training on NIH (ChestX-ray14) dataset

- You can find initial setup and pretrain models in [get_started.md](get_started.md).
We used ImageNet-22K pre-trained model Swin-L with 224x224 resolution for our training.

- If you had problems installing `Apex`, you can install it using conda-forge:
  ```
  conda install -c conda-forge nvidia-apex
  ```

  Additionally, install `numpy`, `pillow`, `pandas`, `scikit-learn` and `scipy` packages with `pip` or `conda`:
  ```
  pip install [package-name]
  ```

- Download NIH (ChestX-ray14) dataset from [Kaggle](https://www.kaggle.com/nih-chest-xrays/data).
Merge images from different folders into one folder. Optionally, you can have different folders for train, validation, and test data.

- For training the model on one gpu run:
  ```
  python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 main.py \
  --cfg configs/swin_large_patch4_window7_224.yaml --resume path/to/pretrain/swin_large_patch4_window7_224_22k.pth \
  --trainset path/to/train_data/ --validset path/to/validation_data/ --testset path/to/test_data/ \
  --train_csv_path configs/NIH/train.csv --valid_csv_path configs/NIH/validation.csv --test_csv_path configs/NIH/test.csv \
  --batch-size 32 [--output <output-directory> --tag <job-tag> --num_mlp_heads 3] > log.txt
  ```

  You can extract validation and test ROC_AUC scores from resulting `log.txt` file.
  
  Note: As mentioned above, if you have all data in one folder, then the `trainset`, `validset` and `testset` arguments would point to the same folder.

  Note: If you want to continue a half-trained model from a checkpoint, you should comment the line specified with "TODO" in `utils.py`.
