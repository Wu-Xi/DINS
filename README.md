# DINS
DINS: Dimension Independent Mixup for Hard Negative Sample in Collaborative Filtering

This is our PyTorch implementation for the paper.


## Environment Requirement

The code has been tested running under Python 3.8.0 and torch 2.0.0.

The required packages are as follows:

- pytorch == 2.0.0
- numpy == 1.22.4
- scipy == 1.10.1
- sklearn == 1.1.3
- prettytable == 2.1.0

## Training

The instruction of commands has been clearly stated in the codes (see the parser function in utils/parser.py). Important argument:

- `alpha`
  - It regulates the overall inclination of the synthesized hard negative examples to approach the positive examples.
- `n_negs`
  - It specifies the size of negative candidate set when using DINS.

#### LightGCN_DINS

```
python main.py --dataset ali --dim 64 --lr 0.001 --batch_size 2048 --gpu_id 0  --pool mean --ns dins --alpha 2 --n_negs 32

python main.py --dataset yelp2018 --dim 64 --lr 0.001 --batch_size 2048 --gpu_id 0  --pool mean --ns dins --alpha 0.3 --n_negs 64

python main.py --dataset amazon --dim 64 --lr 0.001 --batch_size 2048 --gpu_id 0  --pool mean --ns dins --alpha 2.3 --n_negs 32
```

#### NGCF_DINS

```
python main.py --dataset ali --gnn ngcf --dim 64 --lr 0.0001 --batch_size 1024 --gpu_id 0  --pool concat --ns dins --alpha 7 --n_negs 64

python main.py --dataset yelp2018 --gnn ngcf --dim 64 --lr 0.0001 --batch_size 1024 --gpu_id 0  --pool concat --ns dins --alpha 2.5 --n_negs 64

python main.py --dataset amazon --gnn ngcf --dim 64 --lr 0.0001 --batch_size 1024 --gpu_id 0  --pool concat --ns dins --alpha 2.5 --n_negs 64
```


#### MF_DINS

```
python main.py --dataset ali --gnn ngcf --dim 64 --lr 0.0001 --batch_size 1024 --gpu_id 0  --pool concat --ns dins --alpha 9.5 --n_negs 32

python main.py --dataset yelp2018 --gnn ngcf --dim 64 --lr 0.0001 --batch_size 1024 --gpu_id 0  --pool concat --ns dins --alpha 2.5 --n_negs 64

python main.py --dataset amazon --gnn ngcf --dim 64 --lr 0.0001 --batch_size 1024 --gpu_id 0  --pool concat --ns dins --alpha 8 --n_negs 16
