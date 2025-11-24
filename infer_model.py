#!/usr/bin/env python3
"""
Gradio inference app for the trained keyphrase generator.
Usage:
    pip install gradio transformers sentencepiece torch
    python gradio_infer_t5.py --model_dir ./t5_quick
"""
import argparse
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

def load_model(model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    return tokenizer, model

def generate_keywords(text, model_dir, max_len=64, num_beams=5, top_k=10):
    if not text or text.strip() == "":
        return ""
    tokenizer, model = load_model(model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    input_text = "generate keywords: " + text
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k:v.to(device) for k,v in inputs.items()}
    outputs = model.generate(**inputs, max_length=max_len, num_beams=num_beams, early_stopping=True, num_return_sequences=1)
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    items = [i.strip() for i in decoded.replace("\n",";").split(";") if i.strip()]
    return "\\n".join(items[:top_k])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="./t5_quick", help="Directory with trained model and tokenizer")
    parser.add_argument("--title", type=str, default="T5 Keyphrase Generator")
    args = parser.parse_args()

    # Warm load
    load_model(args.model_dir)

    demo = gr.Interface(
        fn=lambda text, model_dir, max_len, num_beams, top_k: generate_keywords(text, model_dir, max_len, num_beams, top_k),
        inputs=[gr.Textbox(lines=8, label="Input text"),
                gr.Textbox(value=args.model_dir, label="Model dir"),
                gr.Slider(10, 128, value=64, step=1, label="Max generated length"),
                gr.Slider(1, 10, value=5, step=1, label="Num beams"),
                gr.Slider(1, 30, value=10, step=1, label="Top K")],
        outputs=gr.Textbox(lines=10, label="Generated keyphrases"),
        title=args.title,
        description="Load a trained T5 model directory and generate keyphrases."
    )
    demo.launch()

if __name__ == "__main__":
    main()
