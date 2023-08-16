import argparse
from model import FLANT5Config, FLANT5
import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer
import time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", type=str, default="base", choices=["small", "base", "large"])
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--num_beams", default=1, type=int)
    parser.add_argument("--max_new_tokens", default=20, type=int)
    args = parser.parse_args()

    checkpoint_name = f"google/flan-t5-{args.model_size}"
    model = FLANT5.from_pretrained(checkpoint_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device);
    model.eval();

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_name)

    out = model.generate(tokenizer=tokenizer,
                    prompt_text=args.prompt,
                    max_new_tokens=args.max_new_tokens,
                    num_beams=args.num_beams)
    print(out)


