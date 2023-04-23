# Databricks notebook source
# models                num layers
#------------------------------
# bert-base-uncased     12
# bert-large-uncased    24
# gpt2                  12
# gpt2-medium           24
# gpt2-large            36

# COMMAND ----------

pip install -U spacy[cuda-autodetect] spacy-transformers

# COMMAND ----------

# DATABASE_NAME = 'ktest.bigTransformerNERSurvey2'
# DATA_PATH = '/dbfs/FileStore/melissa_sample_prodigy/eas_model_04_05_23'

# DATABASE_NAME = 'ktest.bigTransformerNERSurveyPII'
# DATA_PATH = '/dbfs/FileStore/spacy-pretrain/ner_pii6'

DATABASE_NAME = 'ktest.bigTransformerNERSurveyCONLL'
DATA_PATH = '/dbfs/FileStore/CONLL2003/spacy'

params_dict = dict(dbutils.notebook.entry_point.getCurrentBindings())

# COMMAND ----------

HF_model_name = params_dict.get('HF_model_name', 'bert-base-uncased')
setup_function = params_dict.get('setup_function', 'setup_roberta.v0')
is_GPT = True if 'true' == params_dict.get('is_GPT') else False

l = params_dict.get('layers', None)
assert(l is not None)
LAYERS = list(map(int, l.split(',')))
assert(len(LAYERS) > 0)

# COMMAND ----------

import spacy
import spacy_transformers

assert(spacy.require_gpu())

# COMMAND ----------

from spacy.util import registry
from spacy_transformers.layers import TransformerModel
from spacy_transformers.layers.transformer_model import set_pytorch_transformer

from typing import Dict, List, Any, Callable
from thinc.api import Model
from spacy.tokens import Doc
from spacy_transformers.data_classes import FullTransformerBatch, HFObjects

# for debugging
global_transformer = None

@registry.misc("setup_gptneox.v0")
def freeze_layers_and_set_gptneox(model: Model, hf_model: HFObjects):
    global global_transformer
    
    tokenizer   = hf_model.tokenizer
    transformer = hf_model.transformer
    
    # for debugging
    global_transformer = transformer

    # freeze all but the top Nkeep layers:
    nkeep = model.attrs["n_layers_to_keep"]
    n = len(transformer.layers)
    modules = [ transformer.embed_in, *transformer.layers[:(n-nkeep)] ]
    for module in modules:
        for param in module.parameters():
            param.requires_grad = False

    total_params = transformer.num_parameters(only_trainable=False)
    trainable_params = transformer.num_parameters(only_trainable=True)
    print(f"total params: {total_params:,.0f}\ntrainable params: {trainable_params:,.0f}\n={trainable_params/total_params:.4%}")
    
    set_pytorch_transformer(model, hf_model)

@registry.misc("setup_gpt2.v0")
def freeze_layers_and_set_gpt2(model: Model, hf_model: HFObjects):
    global global_transformer
    
    tokenizer   = hf_model.tokenizer
    transformer = hf_model.transformer
    
    # for debugging
    global_transformer = transformer

    # freeze all but the top Nkeep layers:
    nkeep = model.attrs["n_layers_to_keep"]
    n = len(transformer.h)
    modules = [ transformer.wte, transformer.wpe, *transformer.h[:(n-nkeep)] ]
    for module in modules:
        for param in module.parameters():
            param.requires_grad = False

    total_params = transformer.num_parameters(only_trainable=False)
    trainable_params = transformer.num_parameters(only_trainable=True)
    print(f"total params: {total_params:,.0f}\ntrainable params: {trainable_params:,.0f}\n={trainable_params/total_params:.4%}")
    
    set_pytorch_transformer(model, hf_model)

@registry.misc("setup_roberta.v0")
def freeze_layers_and_set_roberta(model: Model, hf_model: HFObjects):
    global global_transformer
    
    tokenizer   = hf_model.tokenizer
    transformer = hf_model.transformer
    
    # for debugging
    global_transformer = transformer

    # freeze all but the top Nkeep layers:
    nkeep = model.attrs["n_layers_to_keep"]
    n = len(transformer.encoder.layer)
    modules = [ transformer.embeddings, *transformer.encoder.layer[:(n-nkeep)] ]
    for module in modules:
        for param in module.parameters():
            param.requires_grad = False

    total_params = transformer.num_parameters(only_trainable=False)
    trainable_params = transformer.num_parameters(only_trainable=True)
    print(f"total params: {total_params:,.0f}\ntrainable params: {trainable_params:,.0f}\n={trainable_params/total_params:.4%}")
    
    set_pytorch_transformer(model, hf_model)

@registry.misc("setup_distilbert.v0")
def freeze_layers_and_set_distilbert(model: Model, hf_model: HFObjects):
    global global_transformer
    
    tokenizer   = hf_model.tokenizer
    transformer = hf_model.transformer
    
    # for debugging
    global_transformer = transformer

    # freeze all but the top Nkeep layers:
    nkeep = model.attrs["n_layers_to_keep"]
    n = len(transformer.transformer.layer)
    modules = [ transformer.embeddings, *transformer.transformer.layer[:(n-nkeep)] ]
    for module in modules:
        for param in module.parameters():
            param.requires_grad = False

    total_params = transformer.num_parameters(only_trainable=False)
    trainable_params = transformer.num_parameters(only_trainable=True)
    print(f"total params: {total_params:,.0f}\ntrainable params: {trainable_params:,.0f}\n={trainable_params/total_params:.4%}")
    
    set_pytorch_transformer(model, hf_model)

@registry.architectures("PartiallyFrozenHFModel.v0")
def partially_frozen_distilbert(
    name: str,
    get_spans: Callable,
    n_layers_to_keep: int,
    set_transformer_method: str,
    tokenizer_config: Dict[str, Any],
    transformer_config: Dict[str, Any] = {},
    mixed_precision: bool = False,
    grad_scaler_config: Dict[str, Any] = {}
) -> Model[List[Doc], FullTransformerBatch]:

    model = TransformerModel(name, get_spans, tokenizer_config, transformer_config, mixed_precision, grad_scaler_config)
    
    model.attrs["n_layers_to_keep"] = n_layers_to_keep
    model.attrs["set_transformer"] = registry.get("misc", set_transformer_method)

    return model

# COMMAND ----------

import tempfile
import srsly
import time
from pathlib import Path

def run_test(HF_model_name, setup_function, layers_to_keep, is_GPT=False, is_GPTNEOX=False):
    
    cfg = spacy.cli.init_config(lang='en', pipeline=['ner'], optimize='accuracy', gpu=True, pretraining=False, silent=False)

    cfg['components']['transformer']['model']['@architectures'] = "PartiallyFrozenHFModel.v0"
    cfg['components']['transformer']['model']['name'] = HF_model_name
    cfg['components']['transformer']['model']['set_transformer_method'] = setup_function
    cfg['components']['transformer']['model']['n_layers_to_keep'] = layers_to_keep
    
    if is_GPT:
        cfg['components']['transformer']['model']['tokenizer_config']['pad_token'] = '<|endoftext|>'
    if is_GPTNEOX:
        cfg['components']['transformer']['model']['tokenizer_config']['pad_token'] = '<|endoftext|>'
        cfg['components']['transformer']['model']['transformer_config']['pad_token_id'] = 0
    
    cfg['corpora']['dev']['path']   = DATA_PATH + '/dev.spacy'
    cfg['corpora']['train']['path'] = DATA_PATH + '/train.spacy'

    nlp = spacy.training.initialize.init_nlp(config=cfg, use_gpu=0)
    
    with tempfile.TemporaryDirectory() as dirpath:
        
        outdir = Path(dirpath)
        t_start = time.time()
        nlp_final = spacy.training.loop.train(nlp=nlp, use_gpu=0, output_path=outdir)
        t_end = time.time()
        
        meta = srsly.read_json(outdir / 'model-best' / 'meta.json')

        f_score, loss = meta['performance']['ents_f'], meta['performance']['ner_loss']
        train_time = (t_end - t_start)

    result = {
        'HF_model': HF_model_name,
        'layers_to_keep': layers_to_keep,
        'NER_f_score': f_score,
        'NER_loss': loss,
        'train_time_hours': train_time / 3600
    }
    
    df = spark.createDataFrame([result])
    df.write.saveAsTable(name=DATABASE_NAME, mode='append')
    
    return result

# COMMAND ----------

print("Running experiment for:")
print(f"model name = {HF_model_name:s}\nsetup function = {setup_function}\nis GPT = {is_GPT}")

# COMMAND ----------

for l in LAYERS:
    results = run_test(HF_model_name=HF_model_name, setup_function=setup_function, layers_to_keep=l, is_GPT=is_GPT)

# COMMAND ----------


