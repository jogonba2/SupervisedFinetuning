# Dataset config
local dataset = {
    "path": "bentrevett/multi30k",
    "task_type": "generation",
    "max_n_test": 1000
};

# Shots args
# Zero-shot runs always by default, so no required to put
# 0 in `shots`. If you want just to run zero-shot experiments
# let this array empty --> `local shots = [];`
local shots = [8, 16, 32];

# In-context examples args
local icl_training_examples = [0, 1, 3];
local icl_inference_examples = [0, 1, 3];

# LoRA configs
local loras = {
    "lora_pca_init": {
        r: 8,
        lora_alpha: 8,
        init_lora_weights: "pca",
        target_modules: ["q_proj"],
        bias: "none",
        lora_dropout: 0.15
    },
    "lora_true_init": {
        r: 8,
        lora_alpha: 8,
        init_lora_weights: true,
        target_modules: ["q_proj"],
        bias: "none",
        lora_dropout: 0.15
    }
};

# Prompting configs
local promptings = {
    "prompting_1": {
        instruction: "Translate the following text from English to German.",
        input_columns: [
            "en"
        ],
        output_column: "de",
        max_input_length: 50,
        max_output_length: 50
    },
};

# Model config
local models = {
    "qwen-0.5b": {
        name: "Qwen/Qwen2-0.5B-Instruct",
        quantization: {
            config_class: "BitsAndBytesConfig",
            args: {
                "load_in_8bit": true
            }
        }
    },
    "llama-3-8b-awq": {
        name: "casperhansen/llama-3-8b-instruct-awq",
        chat_template_from: "meta-llama/Meta-Llama-3-8B-Instruct"
    },
    "llama-3-70b-awq": {
        name: "casperhansen/llama-3-70b-instruct-awq",
        chat_template_from: "meta-llama/Meta-Llama-3-70B-Instruct"
    }
};

# Trainer configs
local trainers = {
    "trainer_5_epoch_1e-4":{
        num_train_epochs: 5,
        learning_rate: 1e-4,
        per_device_train_batch_size: 4,
        gradient_accumulation_steps: 2,
        gradient_checkpointing: true,
        max_seq_length: 512,
        logging_steps: 10,
        output_dir: "/tmp",
    }
};

# Inference config
local inferences = {
    "inference_1": {
        batch_size: 8,
        max_seq_length: 512,
        generation_args: {
            do_sample: false,
            max_new_tokens: 50
        }
    }
};

# Zero-shot configs
local zero_shot_configs = {
    [std.join('_', [model, prompting, trainer, inference]) + "_0_" + icl_inference]: {
        dataset: dataset + {shots: 0}, # Fixed zero-shot in dataset
        model: models[model],
        lora: loras[std.objectFields(loras)[0]], # A random LoRA here, since it is not used
        prompting: promptings[prompting],
        training: {"trainer_args": trainers[trainer]} + {icl_examples: 0}, # Zero training icl always
        inference: inferences[inference] + {icl_examples: icl_inference}
    }
    for model in std.objectFields(models)
    for prompting in std.objectFields(promptings)
    for trainer in std.objectFields(trainers)
    for inference in std.objectFields(inferences)
    for icl_inference in icl_inference_examples
};

# Few-shot configs
local few_shot_configs = {
    [std.join('_', [model, lora, prompting, trainer, inference]) + "_" + shot + "_" + icl_train + "_" + icl_inference]: {
        dataset: dataset + {shots: shot},
        model: models[model],
        lora: loras[lora],
        prompting: promptings[prompting],
        training: {"trainer_args": trainers[trainer]} + {icl_examples: icl_train},
        inference: inferences[inference] + {icl_examples: icl_inference}
    }
    for model in std.objectFields(models)
    for lora in std.objectFields(loras)
    for prompting in std.objectFields(promptings)
    for trainer in std.objectFields(trainers)
    for inference in std.objectFields(inferences)
    for shot in shots
    for icl_train in icl_training_examples
    for icl_inference in icl_inference_examples
};

{
    "configs": zero_shot_configs + few_shot_configs
}