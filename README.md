## LaGAN

LaGAN's implementation is based on [U-GAT-IT](https://arxiv.org/abs/1907.10830)'s [official implementation](https://github.com/znxlwm/UGATIT-pytorch).

### Dependencies

#### Ensure unzip is installed
**unzip** is required to unzip datasets via the setup tool.
On many UNIX systems, the dependency can be installed by executing `apt-get install unzip`.

#### Preferred dependency installation
The preferred way to install Python dependencies for this project is via **pipenv**,
a tool for virtual environment and package management.
There are multiple ways to set up **pipenv**, perhaps the easiest is `pip install pipenv`.
By executing `pipenv install`, a new virtual environment (for this project) will be created and dependencies will be installed.

#### Alternative dependency installation
Alternatively, it is possible to produce `requirements.txt` from the **pipenv** configuration
by executing `pipenv requirements > requirements.txt`. This will produce requirements.txt.
Then, we can install all dependencies using **pip**, `pip install -r requirements.txt`.
However, installing in this way may not work, since some dependencies might be impossible to resolve given the currently installed dependencies in the environment where **pip** is executed.

## Dataset preparation
For training to translate source to target domain, it is necessary to (re)organize the dataset to match the folder structure below. 
The suffix 'A' corresponds to the source domain, 'B' corresponds to the target domain.
```
├── dataset
   └── DATASET_NAME
       ├── trainA
           ├── xxx.jpg (name, format doesn't matter)
           ├── yyy.png
           └── ...
       ├── trainB
           ├── zzz.jpg
           ├── www.png
           └── ...
       ├── valA
           ├── uuu.jpg
           ├── iii.png
           └── ...
       ├── valB
           ├── ooo.jpg
           ├── jjj.png
           └── ...
       ├── testA
           ├── aaa.jpg 
           ├── bbb.png
           └── ...
       └── testB
           ├── ccc.jpg 
           ├── ddd.png
           └── ...
```

Assuming the datasets are set up in the **uda** code folder, it is possible to automatically (re)organize 
the dataset to match the structure above. The example below illustrated how this can be done.
In the example below, we are translating healthy to rust from the source domain (Plant Village).

Firstly, navigate to the **uda** folder. By executing the following script, the dataset in **lagan** code folder 
is created, matching the expected format.
```bash
DATASET="plant-village-healthy-to-plant-village-rust"
LAGAN_PATH='{path to lagan code}'

chmod +x scripts/datasets/setup_cycle_gan_dataset.py
PYTHONPATH=${PWD} ${PWD}/scripts/datasets/setup_apples.py --domains plant-village --img_size 286
PYTHONPATH=${PWD} ${PWD}/scripts/datasets/setup_cycle_gan_dataset.py --datasets-path $LAGAN_PATH/dataset \
                                                                     --dataset-name $DATASET \
                                                                     --source-train data/apples/plant-village/train/healthy \
                                                                     --source-val data/apples/plant-village/val/healthy \
                                                                     --source-test data/apples/plant-village/test/healthy \
                                                                     --target-train data/apples/plant-village/train/rust \
                                                                     --target-val  data/apples/plant-village/val/rust \
                                                                     --target-test data/apples/plant-village/test/rust
```

### Training

In general, to train the model we execute
```bash
python cli.py --phase train \
              --dataset $DATASET \
              --img_size 256 \
              --batch_size $BATCH_SIZE \
              --display_freq $DISPLAY_FREQ \
              --eval_freq $EVAL_FREQ \
              --save_freq $SAVE_FREQ \
              --iters $ITERS \
              --num_bottleneck_blocks $NUM_BOTTLENECK_BLOCKS \
              --nce_weight $NCE_WEIGHT \
              --nce_layers $NCE_LAYERS \
              --lr $LR \
              --iters $ITERS
```
The following concrete example is training default LaGAN to translate from healthy to rust,
assuming Plant Village apples.

```bash
SEED=269902365
DATASET="plant-village-healthy-to-plant-village-rust"

CKPT_100K="iter_0100000"
CKPT_150K="iter_0150000"
CKPT_200K="iter_0200000"
CKPT_250K="iter_0250000"
CKPT_SMALLEST_VAL_FID="smallest_val_fid"

DISPLAY_FREQ=1000
EVAL_FREQ=10000
SAVE_FREQ=50000
ITERS=250000

BATCH_SIZE=1
LR=0.0001
NUM_BOTTLENECK_BLOCKS=9
NCE_WEIGHT=10.0
NCE_LAYERS="0,2,3,4,8"

python cli.py --phase train \
              --dataset $DATASET \
              --img_size 256 \
              --batch_size $BATCH_SIZE \
              --display_freq $DISPLAY_FREQ \
              --eval_freq $EVAL_FREQ \
              --save_freq $SAVE_FREQ \
              --iters $ITERS \
              --num_bottleneck_blocks $NUM_BOTTLENECK_BLOCKS \
              --nce_weight $NCE_WEIGHT \
              --nce_layers $NCE_LAYERS \
              --lr $LR \
              --iters $ITERS
```
When training, the **results** folder will be created, containing training logs and checkpoints,
grouped by ```$DATASET```. The checkpoints are stored in ```./results/$DATASET/model```. The logs will include loss logs, learning rate scheduler logs and the image translations obtained during the training in various stages.

### Dataset translation
To translate the dataset, it is necessary to specify the checkpoint, the number of bottleneck 
blocks and the layers from which patches are sampled. This is necessary to properly configure the model and load the appropriate checkpoint. The checkpoint ```$CKPT.pt``` must be present in ```./results/$DATASET/model```.

To translate entire source domain to the target domain, we execute 
```bash
python cli.py --phase translate \
              --dataset $DATASET \
              --num_bottleneck_blocks $NUM_BOTTLENECK_BLOCKS \
              --nce_layers $NCE_LAYERS \
              --ckpt $CKPT.pt
```
The translations will be stored in ```./translations/$DATASET```.

The following concrete example is translating from healthy to rust,
assuming Plant Village apples and default model configuration.

```bash
DATASET="plant-village-healthy-to-plant-village-rust"
NUM_BOTTLENECK_BLOCKS=9
NCE_LAYERS="0,2,3,4,8"
CKPT="smallest_val_fid"

python cli.py --phase translate \
              --dataset $DATASET \
              --num_bottleneck_blocks $NUM_BOTTLENECK_BLOCKS \
              --nce_layers $NCE_LAYERS \
              --ckpt $CKPT.pt
```

### Image Quality Evaluation
To evaluate image quality, we use [torch-fidelity](https://github.com/toshas/torch-fidelity).
For example, to compute the test FID, we execute
```bash
fidelity --gpu 0 --fid --input1 dataset/$DATASET/testB --input2 translations/$DATASET/$CKPT/test
```