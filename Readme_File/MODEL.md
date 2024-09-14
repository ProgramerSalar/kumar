

* [build_transformer](#build_transformer)
* [LayerNormalization](#layerNormalization)
* [FeedForwardBlock](#FeedForwardBlock)
* [InputEmbeddings](#InputEmbeddings)
* [PositionalEncoding](#PositionalEncoding)
* [MultiHeadAttentionBlock](#MultiHeadAttentionBlock)
* [ProjectionLayer](#ProjectionLayer)
* [ResidualConnection](#ResidualConnection)
* [EncoderBlock](#EncoderBlock)
* [Encoder](#Encoder)
* [DecoderBlock](#DecoderBlock)
* [Transformer](#Transformer)





<!-- , , , , , , , , , , ,  -->

## build_transformer

- if you are import the model of the Transformer then used to this import 
```
# import the transformer model 
>>> from PSTansformer.model import build_transformer

# how to used this transformer model 
>>> build_transformer(
        vocab_src_len=vocabulary_source_length,   # vocabulary source length of sentence like tokeinzer source length of 
        vocab_tgt_len=vocabulary_target_length,    # same for the target language 
        src_seq_len=config["seq_len"],     # source language  length of you sentence like 350 
        tgt_seq_len=config['seq_len'],     # target language length of you sentence same as source length
        d_model=config['d_model']        # dimension model your language like 512
)
```

## LayerNormalization

```
>>> from kumar.Transformer.model import LayerNormalization

LayerNormalization(
    features:int,   # 512 features 
    eps:float=10**-5
)
```


## FeedForwardBlock

```

>>> from kumar.Transformer.model import FeedForwardBlock

FeedForwardBlock(
    d_model:int,       # 512 features 
    d_ff: int,         # 2048 hidden layer 
    dropout: float     # 0.1
)
```


## InputEmbeddings

```
>>> from kumar.Transformer.model import InputEmbeddings

InputEmbeddings(
    d_model:int,     # dimension model of your data Ex: 512
    vocab_size       # vacabulary size of you data Ex:  1000
)
```


## PositionalEncoding

```
>>> from kumar.Transformer.model import PositionalEncoding 

PositionalEncoding(
    d_model: int,      # 512
    seq_len: int,      # 2000
    dropout: float     # 0.1
)
```


## MultiHeadAttentionBlock

```
>>> from kumar.Transformer.model import MultiHeadAttentionBlock

MultiHeadAttentionBlock(
    d_model: int,          # 512
    h: int,                # 8
    dropout: float         # 0.1
)
```




## ProjectionLayer

```
>>> from kumar.Transformer.model import ProjectionLayer 


ProjectionLayer(
    d_model: int,   
    vocab_size:int

)
```

## ResidualConnection

- please define the ```LayerNormalization``` Function Before this used 

```
>>> from kumar.Transformer.model import ResidualConnection

ResidualConnection(
    features: int,
    dropout
)
```


## EncoderBlock

- you might define ```MultiHeadAttentionBlock``` and ```FeedForwardBlock```

```
>>> from kumar.Transformer.model import EncoderBlock

EncoderBlock(
    features:int,
    self_attention_block: MultiHeadAttionBlock,
    feed_forward_block: FeedForwardBlock,
    dropout: float
)
```


## Encoder

- please define the ```LayerNormalization``` Function Before this used 


```
>>> from kumar.Transformer.model import Encoder

Encoder(
    features: int,
    layers: nn.ModuleList
)
```


## DecoderBlock

- you might define ```MultiHeadAttentionBlock``` and ```FeedForwardBlock```


```
>>> from kumar.Transformer.model import DecoderBlock

DecoderBlock(
    features: int,
    self_attention_block: MultiHeadAttentionBlock,
    cross_attention_block: MultiHeadAttentionBlock,
    feed_forward_block: FeedForwardBlock,
    dropout: float
)
```



## Decoder

- please define the ```LayerNormalization``` Function Before this used 


```
>>> from kumar.Transformer.model import Decoder

Decoder(
    features:int,
    layers: nn.ModuleList
)
```


## Transformer

- you might define ```Encoder``` and ```Decoder``` and ```PostionalEncoding``` and ```ProjectionLayer```
- please define the ```InputEmbeddings``` Function Before this used 




```
>>> from kumar.Transformer.model import Transformer

Transformer(
    encoder: Encoder,
    decoder: Decoder,
    src_embed: InputEmbeddings,
    tgt_embed: InputEmbeddings,
    src_pos: PostionalEncoding,
    tgt_pos: PostionalEncoding,
    projection_layer: ProjectionLayer
)
```