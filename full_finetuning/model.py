from transformers import AutoModelForCausalLM, AutoConfig

def load_text_generation_model(
    model_type, 
    train_type,
    output_attentions=False,
):
    config = AutoConfig.from_pretrained(
        model_type, output_attentions=output_attentions,
    )
    model = AutoModelForCausalLM.from_pretrained(model_type, config=config)

    return model