from transformers import AutoModelForSeq2SeqLM, AutoConfig
from peft import (
    get_peft_model,
    LoraConfig, AdaLoraConfig, PrefixTuningConfig,
)

def load_text_generation_model(
    model_type, 
    train_type, 
    output_attentions=False,
):

    config = AutoConfig.from_pretrained(
    model_type, output_attentions=output_attentions,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(model_type, config=config)

    return model