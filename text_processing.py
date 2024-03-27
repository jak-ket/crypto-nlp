import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from data_utils import load_reddit_data
import torch
from time import sleep

bnb_config = BitsAndBytesConfig(
load_in_4bit=True,
bnb_4bit_use_double_quant=True,
bnb_4bit_quant_type="nf4",
bnb_4bit_compute_dtype=torch.bfloat16
)


def mistral_relevant_text_identification(df: pd.DataFrame, text_column: str):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Quantized model
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", quantization_config=bnb_config, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    
    # We don't use a system prompt as Mistral does not support it
    messages = [
        {
            "role": "user",
            "content": "You read a text and reply only by yes or know. Your job is to assess if the text is about bitcoin trading. If you are unsure reply no.",
        },
        {
            "role": "assistant",
            "content": "Of course! Provide me the text",
        },
        None
    ]
    

    tagging = []

    for k, text in enumerate(df[text_column]):
        messages[2] = {"role": "user", "content": text}
        
        encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

        model_inputs = encodeds.to(device)

        generated_ids = model.generate(model_inputs, max_new_tokens=5, do_sample=True)
        decoded = tokenizer.batch_decode(generated_ids)[0]

        if "es" in decoded[-5:]:
            tagging.append(1)
        else:
            tagging.append(0)
            
        if k == 1000:
            break

    tagging = tagging + [np.nan] * (len(df) - len(tagging))

    df["mistral_tagging"] = tagging

    return df

if __name__ == "__main__":
    
    df = load_reddit_data()
    df = mistral_relevant_text_identification(df, "selftext")
    
    df.to_csv("mistral_tagging.csv", index=False)
