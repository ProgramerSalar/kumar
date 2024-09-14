
# Reference 

* [MODEL](https://github.com/ProgramerSalar/kumar/blob/master/Readme_File/MODEL.md)
* [BilingualDataset](#BilingualDataset)
* [train_model](#train_model)
* [latest_weights_file_path](#latest_weights_file_path)
* [get_weights_file_path](#get_weights_file_path)
* [Diffusion](https://github.com/ProgramerSalar/kumar/blob/master/Readme_File/DIFFUSION.md)


## BilingualDataset

- if you import the Tensor dataset function, which is convert the tensor data from raw data 
```
# import the Tensor dataset Function
>>> from PSTansformer.dataset import BilingualDataset

# how to used this Tensor dataset which is convert to the Tensor of the row data 
>>> BilingualDataset(
        ds=train_dataset_raw,   # raw dataset like='Ram eats mango'
        tokenizer_src=tokenizer_source,  # source language tokenizer 
        tokenizer_tgt=tokinzer_target,   # target language tokenizer
        src_lang=config['lang_src'],      # source language like engish
        tgt_lang=config['lang_tgt'],     # target language like Hindi
        seq_len=config['seq_len'])      # sequence length like 350
```

## train_model

- how to used the train model 
- define the ```get_config```  function which is used in the ```train_model(config)```

```


def get_config():
    return {
        "batch_size": 8,
        "num_epochs": 20,
        "lr": 10**-4,
        "seq_len": 350,
        "d_model": 512,
        "datasource": 'opus_books',
        "lang_src": "en",
        "lang_tgt": "it",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": "latest",
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel"
    }
```

```
>>> from kumar.Transformer.train import train_model 

    config = get_config()
    train_model(config)
```

## latest_weights_file_path

- Find the latest weights file in the weights folder
- define the ```get_config```  function which is used in the ```latest_weight_file_path```
```


def get_config():
    return{
        "datasource": 'opus_book',    # Name of data
        "model_folder": "weights",    # put the name in the model folder 
        "model_basename": "tmodel_"   # put the name in the model basename or file name 
    }
```

```
>>> from kumar.Transformer.config import latest_weights_file_path

config = get_config()
latest_weights_file_path(config)
```


## get_weights_file_path

- GEt the weight file Because if model training are break then continoue the model training 

-  ```get_weights_file_path``` function GET the weight file, Before Function i was create the weight file where have the weights of train data 

- model training time if break the training ThEN 
    - if you are not used ```get_weights_file_path``` then traing start in 1st epochs and put the weight in again

    - If you are used ```get_weights_file_path``` then continoue training, starting of breaking point of training 


```
def get_config():
    return{
        "datasource": 'opus_book',    # Name of data
        "model_folder": "weights",    # put the name in the model folder 
        "model_basename": "tmodel_"   # put the name in the model basename or file name 
    }
```


```
>>> from kumar.Transformer.config import get_weights_file_path

get_weights_file_path(config, epoch:str)
```
