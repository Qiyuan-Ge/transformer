# Transformer

<div align=center>
<img src="https://user-images.githubusercontent.com/53368178/197381681-7b95ed1c-f921-4ae2-87d0-32c7ec358827.png">
</div>

## Usage
````
model = Transformer(src_vocab_size=512, tgt_vocab_size=512, embedding_dim=512, d_model=512, ffn_num_hiddens=2048, num_heads=8, num_blocks=6, dropout=0.1)
src = torch.tensor([[1, 2, 0, 0], [2, 4, 5, 6]])
tgt = torch.tensor([[3, 4, 0], [6, 7, 8]])
mask = create_mask(src, tgt)
out = model(src, tgt, mask[0], mask[1], mask[2])
````

## Citation
````
@misc{vaswani2017attention,
    title   = {Attention Is All You Need},
    author  = {Ashish Vaswani and Noam Shazeer and Niki Parmar and Jakob Uszkoreit and Llion Jones and Aidan N. Gomez and Lukasz Kaiser and Illia Polosukhin},
    year    = {2017},
    eprint  = {1706.03762},
    archivePrefix = {arXiv},
    primaryClass = {cs.CL}
}
````



