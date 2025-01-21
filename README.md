<!---
Copyright 2024 Symanto

Licensed under the CC BY-NC-ND 4.0 License

You must give appropriate credit, provide a link to the license, and indicate if changes were made.
You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
You may not use the material for commercial purposes.
If you remix, transform, or build upon the material, you may not distribute the modified material.
You are free to copy and redistribute this material as it is in any medium or format
You may obtain a copy of the License at

    https://creativecommons.org/licenses/by-nc-nd/4.0/
-->

<h1 align="center">👉 Supervised Finetuning </h1> 
<p align="center">
    <a href="LICENSE">
        <img alt="license" src="https://img.shields.io/badge/license-CC_BY_NC_ND_4.0-green">
    </a>
    <a href="CODE_OF_CONDUCT.md">
        <img alt="Contributor Covenant" src="https://img.shields.io/badge/Contributor%20Covenant-v2.0-green">
    </a>
    <img alt="Python version" src="https://img.shields.io/badge/Python-3.10-blue">
</p>

<h3 align="center">
    <p><b>Train and evaluate supervised-finetuned LLMs in zero and few-shot settings</b></p>
</h3>

# 📖 Introduction 
---
👉 **SupervisedFinetuning** is intended to experiment with LoRA adapters for the supervised finetuning of HuggingFace-available LLMs in zero and few-shot settings.
This framework aims to replace the use of evaluation pipelines involving [LM-Eval](https://github.com/EleutherAI/lm-evaluation-harness) and custom internal scripts for supervised finetuning. Although you still can use LM-Eval for evaluating your models trained with **SupervisedFinetuning** if you want!

Actually, 👉**SupervisedFinetuning** supports:

- Few-shot training through `SFTrainer` (proper loss masking, prompting, etc. for free!)
- Generation and fast logit-based classification.
- In-Context Examples (ICE) in training and inference.
- Automatic chat-prompt templating, unifying most of the models in the HF hub.
- Flexibility to manage input/output/ICE lengths.
- Multiple input columns.
- Bitsandbytes and AWQ quantized models.
- Custom LoRA initializations and flexibility to extend them.
- Calibration methods like contextual, domain, and batch calibration.
- Evaluation and results aggregation.
- Single/Multiple experiments using JSON and JSONNET.

# 🔧 Installation
---
Install the requirements as:

```bash
pip install -r requirements.txt
```

If you want to contribute to this repo, please install the dev-requirement and use dev-tools:

```bash
pip install -r dev_requirements.txt
```

# 🔴 Endpoints
---
👉**SupervisedFinetuning** exposes 5 endpoints:
- **train**: trains a model given a JSON configuration file.
- **predict**: computes predictions with a model given a JSON configuration file.
- **evaluate**: evaluates the predictions of a model and stores them in a JSON file.
- **run-experiment**: runs a train + evaluate single experiment given a JSON configuration file.
- **run-experiments**: runs single/multiple experiments, by calling `run-experiment` through `subprocess`, thus ensuring the GPU memory is free after each experiment. This endpoint requires a JSONNET configuration file.

Before continuing, please, take a look to configuration files in [etc/configs/json](etc/configs/json) and [etc/configs/jsonnet](etc/configs/jsonnet). To know more about configuration files, please, read the **About configuration files** section in this README.

## 🏋️ Train endpoint
The `train` endpoint requires to define, at least, the **dataset**, **model**, **prompting**, **lora** and **training** fields in your JSON config. You can run this endpoint as:

```bash
python -m sft.cli train etc/configs/json/tweet_sentiment_qwen.json
```

The checkpoint of the trained model will be saved in the `output_dir` you specified in the `trainer_args` of the **training** field in the config.

## 🤖 Predict endpoint
The `predict` endpoint requires to define, at least, the **dataset**, **model**, **prompting**, and **inference** fields in your JSON config. You can run this endpoint as:

```bash
python -m sft.cli predict etc/configs/json/tweet_sentiment_qwen.json --results-path predictions
```

The predictions will be stored in disk, in the specified `results_path` (which is "predictions" by default), as a TSV file including all the columns of the test dataset, plus a new column called "predictions" containing the prediction for each sample.

## ✅ Evaluate endpoint
The `evaluate` endpoint requires to define, at least, the **dataset**, **model**, **prompting**, and **inference** fields in your JSON config. You can run this endpoint as:

```bash
python -m sft.cli evaluate etc/configs/json/tweet_sentiment_qwen.json --results-path results
```

The evaluation results will be stored in disk, in the specified `results_path` (which is "results" by default), as a JSON file including the random seed, the configuration, and the evaluation metrics computed by the [Symanto's metrics in HuggingFace](https://huggingface.co/symanto), which depend on the task type.

Of course, you can use LM-Eval instead of this endpoint, using the models you train with the `train` endpoint. LM-Eval is automatically installed with the `requirements.txt` file, so feel free to use it.

## 🧪 Run experiment endpoint
The `run-experiment` endpoint requires to define all the fields described above in your JSON config. However, there is one exception: you can avoid defining the **lora** field when using models in a zero-shot way. Anyway, this is optional, since if `shots=0` is specified in the **dataset** field, 👉**SupervisedFinetuning** will not include LoRA adapters in the model. You can run this endpoint as:

```bash
python -m sft.cli run-experiment etc/configs/json/tweet_sentiment_qwen.json --results-path results
```

As in the **train** endpoint, the checkpoint will be saved in disk if you pass the proper arguments to the `trainer_args`. Also, the results will be stored in disk in the same manner than in the **evaluate** endpoint.

## 🏭 Run experiments endpoint
The `run-experiments` endpoint is a wrapper of `run-experiment`, but allows to run multiple experiments defined in a JSONNET file. This endpoint is useful when experimenting with different shots, in-context examples, lora configs, prompts, models, etc, in a single run. Then, you need to define a JSONNET configuration file from which all the single configs are built. This endpoint also ensures that the memory is properly released after each experiment, so, if all your experiments fit in GPU memory, should not be OOM issues. You can run this endpoint as:

```bash
python -m sft.cli run-experiments etc/configs/jsonnet/tweet_sentiment.jsonnet --results-path results
```

In the same way than `run-experiment` each model could be stored in disk, and all the results will be stored in disk. At the end, the results are also aggregated in a JSON and a Markdown file for an easy comparison of the results.

# ⚙️ About configuration files
---
The configuration files for the endpoints expecting JSON files could contain, at most, six outer keys: **dataset**, **model**, **lora**, **prompting**, **training**, and **inference**. Of course, these keys depend on the endpoint, but nothing prevents you to define all of them, since each endpoint will use those that are required to work. For instance, the endpoint **train** will not use the **inference** key, or the endpoint **evaluate** will not use the **lora** (since the model is already defined and loaded as it is) or **training** keys. Here is a summary of the keys and the values that can be used in JSON configs:

- **dataset**: specifies the dataset arguments
  - **path** (str): path to the dataset. The dataset is expected to have `train` and `test` splits.
  - **shots** (int): number of training shots to sample from the dataset
  - **max_n_test** (int): number of samples to use for evaluating with the test set
  - **args** (dict): arguments to be padssed to `load_dataset` or `load_from_disk`
  - **task_type** (str): either `generation` or `classification`

- **model**: specifies the arguments for the model
  - **name** (str): path to the model (HF hub or local).
  - **quantization** (dict): must contain `config_class` to specify the quantization configuration class (e.g., `BitsAndBytesConfig` or `GPTQConfig`) and `args`, to specify the arguments to be passed to the [config class object](https://huggingface.co/docs/transformers/main_classes/quantization).
  - **model_args** (dict): other model arguments to be passed to the [from_pretrained](https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.from_pretrained) method. Additionally, there are two arguments that are used to load model differently:
    - **parallelize** (bool): decides whether split weights acros available gpus.
    - **no_split_module_classes** (Optional[list[str]]): A list of layer class names that should never be split across device (for instance any layer that has a residual connection).

  - **chat_template_from** (str): the name of the other model whose chat template will be used by the model `name`. If empty, the chat template of the model `name` will be used. This is especially useful when third-party models have some bug in the template, but the original model already fixed it, e.g., the chat template of `casperhansen/llama-3-70b-instruct-awq` is fixed by `meta-llama/Meta-Llama-3-70B-instruct`.

- **lora**: specifies the arguments expected by [LoraConfig](https://huggingface.co/docs/peft/package_reference/lora#peft.LoraConfig). When using custom initializations, you have to specify the name of one initialization in the [registry](sft/initialization/initializers/__init__.py) into the `init_lora_weights` parameter, and 👉**SupervisedFinetuning** will do the rest.

- **prompting**: specifies the arguments for prompting
  - **instruction** (str): the instruction of the prompt (role system).
  - **input_columns** (list[str]): the input columns to be used as input in the prompt.
  - **output_column** (str): the output column to be included in the prompt during training and used as reference in the evaluation.
  - **max_input_length** (int): the maximum length of the input texts. Each input text will be truncated to `max_input_length` before being included into the prompt.
  - **max_output_length** (int): the maximum length of the output text. The output text will be truncated to `max_output_length` before being included into the prompt but not in the evaluation when used as reference.

- **training**: specifies the arguments for training
  - **icl_examples** (int): number of in-context examples to be included in the prompt when training. If `task_type` is `generation`, `icl_examples` random examples will be included. If `task_type` is `classification`, `icl_examples` random examples **per-label** will be included.
  - **trainer_args** (dict): arguments to be passed to the [SFTrainer](https://huggingface.co/docs/trl/sft_trainer#trl.SFTTrainer).

- **inference**: specifies the arguments for inference
  - **icl_examples** (int): number of in-context examples to be included in the prompt when doing inference. If `task_type` is `generation`, `icl_examples` random examples will be included. If `task_type` is `classification`, `icl_examples` random examples **per-label** will be included.
  - **batch_size** (int): batch size in inference.
  - **max_seq_length** (int): maximum length of the whole prompt. If the prompt is longer, it will be truncated to `max_seq_length` tokens.
  - **generation_args** (dict): arguments to be passed to the HF [generate](https://huggingface.co/docs/transformers/v4.41.3/en/main_classes/text_generation#transformers.GenerationMixin.generate) method, only for `generation` tasks.
  - **tokenizer_name** (str): path to the model of which we want the tokenizer to be included in the evaluation metrics of BLEU and ROUGE (generation). This is specific for languages such as arabic or chinese where the default tokenizers of the corresponding metrics are not working as desired. For example, for arabic, we are passing the following path "aubmindlab/bert-base-arabert". This argument is Optional, it can be removed if wanted from the config file.
- **evaluation**: specifies the arguments for the different evaluation metrics, these kw_args are included directly in evaluation. More args can be included, however they must coincide with the metrics you want to include these args.
  - **tokenizer_name** (str): tokenizer to use in metrics such as BLEU or ROUGE.

Now that you are familiar with the configuration's fields, take a look again to the configuration files in [etc/configs/json](etc/configs/json) and [etc/configs/jsonnet](etc/configs/jsonnet).

# 👶 Defining new LoRA initializations
---
In few-shot training, a good initialization of the adapter's weights is (thought to be) essential for better and faster convergence. 👉**SupervisedFinetuning** makes this easy, assuming the LoRA weights are initialized from the weight matrices being adapted. To do this, you have to implement a new module and function in [sft/initialization/initializers](sft/initialization/initializers) with the following signature:

```python
from torch import nn

def f(
    layer: nn.Module, rank: int, dtype: torch.dtype = torch.bfloat16
) -> dict[str, torch.Tensor]:
    ...
    return {"lora_A": Tensor, "lora_B": Tensor}
```

which takes a base layer from the (potentially quantized) model, and returns a dictionary with two keys: `lora_A`, a tensor to initialize the A matrix of LoRA and `lora_B`, another tensor to initialize the B matrix of LoRA.

When implementing custom initializations for quantized models, it is highly recommended to dequantize the adapted layer weights before doing any computation. For this, you can use the `dequantize` function in [sft/initialization/utils.py](sft/initialization/utils.py). Currently, this function allows to dequantize int8 and AWQ quantized layer weights.

If you implement your custom initialization, you need to register it in the `initializer_registry` mapping in [sft/initialization/initializers/__init__.py](sft/initialization/initializers/__init__.py). Then, you can define the name of your custom initialization in the `lora` field of your JSON/JSONNET config, and 👉**SupervisedFinetuning** will initialize the LoRA matrices accordingly.

Right now, 👉**SupervisedFinetuning** implements three custom inits:

- **pca**: gets the `lora_rank` eigenvectors with highest eigenvalue from the adapted weights, and uses them to initialize the LoRA A matrix. LoRA B matrix is initialized to 0 to keep the original training dynamics.

- **umap**: reduces dimensionality of the adapted weights to `lora_rank` using UMAP to initialize the LoRA A matrix. LoRA B matrix is initialized to 0 to keep the original training dynamics.

- **pissa**: simplification from https://arxiv.org/pdf/2404.02948, without the residual matrices from SVD and not sharing the random A and B matrices across the model.

# 📚 Literature
---
LoRA adapters have not been explored in the literature for few-shot settings. The most similar works I've found in the literature (blogs and papers) are:

- https://arxiv.org/pdf/2305.14045v1 -> "when LoRA is effective over full-finetuning, in few-shot settings is relatively unexplored in literature, and we leave the additional analysis to future work" + cases where ICL outperforms LoRA (see table 5).

- https://arxiv.org/pdf/2307.13269 -> With 5 examples, ICL and LoRA match performance on a wide set of tasks (see table 1).

- https://lightning.ai/pages/community/lora-insights/ -> LoRA underperforming base 7b model in arithmetic, MMLU, and causation using 50k training samples.

- https://arxiv.org/pdf/2404.02948 -> try to improve slow convergence of LoRA through principal singular vectors.


# 🚫 Limitations
---
- **LoRA**: only LoRA adapters are supported now, as being the reference across the literature.

- **Large label sets and verbalizations**: classifying (i) large label sets and (ii) long label verbalizations (>> number of tokens) can be time exhaustive, since the logit-based classification algorithm needs to iterate over all the token labels. Despite it is implemented reusing the `past_key_values` in an efficient way, the cost is O(|L|x|T|) (L=#labels, T=#tokens), being the all the samples in a batch processed in parallel. This is a current limitation of LLMs when logit-based classification is required.

- **Multi-label classification**: there is no way (to my knowledge) to use LLMs for multi-label classification (using logit-based approaches). Despite in inference you can predict each label independently and you can have an independent probability for each one, it is not clear how to prepare training samples.

- **Structured outputs**: there is no a good way to properly manage structured outputs like JSON using HF LLMs. We can frame it as a `generation` task and specify the format in the instruction, but there is no guarantee of obtaining the proper structure.

# 🗹 TODO List
---
- [X] Test dequantize in AWQ and bitsandbytes
- [X] Test Qwen, Qwen bitsandbytes, LLama-3-70b-awq and LLama-3-8b quantized
- [X] Test all the previous models in zero-shot and few-shot
- [X] Test generation
- [X] Test in classification (as generation, no logits involved at this moment)
- [X] Use collator and dataloader to reduce amount of padding at inference
- [X] Save reports
- [X] Script to run batch of experiments
- [X] Add logit-based classification
- [X] Test logit-based classification
- [X] Add typer in cli
- [X] Add ICL in training and inference
- [X] Add # of ICL examples in training and inference in the results file name
- [X] Support multiple input columns
- [X] Prepare clean configs
- [X] Add clean function to avoid breaking the prompt formats
- [X] Refactor prompting for inference and training, both of them are the same now.
- [X] Add script to aggregate results
- [X] Test all before pushing
- [X] Move the repo to "Dev.SupervisedFinetuning"
- [X] JSONNET configs
- [X] README
- [ ] Support other adapters and keep the `initialization` functionality for them.

# Contribute
Please, use `dev-tools` to contribute to this repo.
