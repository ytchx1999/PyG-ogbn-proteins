# GEN + FLAG + node2vec
This is an improvement of the  [GEN (DeepGCNLayer)](https://github.com/rusty1s/pytorch_geometric/blob/master/examples/ogbn_proteins_deepgcn.py)  model, using the FLAG method and node2vec embedding. 

Our paper is available at [https://arxiv.org/pdf/2105.08330.pdf](https://arxiv.org/pdf/2105.08330.pdf).

### ogbn-products

+ Check out the model： [GEN (DeepGCNLayer)](https://github.com/rusty1s/pytorch_geometric/blob/master/examples/ogbn_proteins_deepgcn.py)
+ Check out the FLAG method：[FLAG](https://arxiv.org/abs/2010.09891)
+ Check out node2vec model：[node2vec](https://arxiv.org/abs/1607.00653)

#### Improvement Strategy：

+ add FLAG method
+ add node2vec embedding

#### Environmental Requirements

+ pytorch == 1.8.1
+ pytorch_geometric == 1.6.3
+ ogb == 1.3.0

#### Experiment Setup：

1. Generate node2vec embeddings, which save in `proteins_embedding.pt`

   ```bash
   python node2vec_proteins.py
   ```

2. Run the real model

   + **Let the program run in the foreground.**

   ```bash
   python gen_em_flag.py
   ```

   + **Or let the program run in the background** and save the results to a log file.

   ```bash
   nohup python gen_em_flag.py > ./gen_em_flag.log 2>&1 &
   ```

#### Detailed Hyperparameter:

```bash
num_layers = 28
hidden_dim = 64
dropout = 0.1
lr = 0.01
runs = 10
epochs = 200
learn_t = 0.1
block = 'res+'
step-size = 1e-4
m = 3
```

node2vec:

```bash
embedding_dim = 64
lr = 0.01
batch_size = 256
walk_length = 80
epochs = 5
```

#### Result:

```bash
All runs:
Highest Train: 91.34 ± 0.91
Highest Valid: 86.56 ± 0.37
  Final Train: 91.34 ± 0.91
   Final Test: 82.51 ± 0.43
```

| Model                 | Test Accuracy   | Valid Accuracy  | Parameters | Hardware          |
| --------------------- | --------------- | --------------- | ---------- | ----------------- |
| GEN + FLAG + node2vec | 0.8251 ± 0.0043 | 0.8656 ± 0.0037 | 487436     | Tesla V100 (32GB) |

