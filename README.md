# Nano FLAN T5

This is an easy-to-read implementation of FLAN-T5. I used Hugginface's excellent [implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py) for reference, and also borrowed code snippets from it (such as for computing positional embeddings). I also used ideas from Andrej Karpathy's instructive [NanoGPT](https://github.com/karpathy/nanoGPT). I wrote this repo just for fun and self-education, and without the two references above, this repo wouldn't exist. All error are mine.

**The good**
* I implemented attention using PyTorch's [`scaled_dot_product_attention`](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) which uses [FlashAttention](https://arxiv.org/abs/2205.14135) under the hood. That said, I didn't see any speedup vs. the HF implementation. This is because (to my knowledge) current CUDA implementations used by SDPA don't support additive biases to attention logits (which we need for T5 arch's positional 'embeddings' to work). That said, it's a matter of time before it changes. Also I think there are some other implementations which do support and I'll explore them next.
* I implemented beam search decoding in the `generate` function. My end-to-end inference speed matches HF's. I also use KV cache (for both self/cross-attn layers) to speed up decoding by reusing cached key-value. Using same hyperparameters, I compared my outputs vs. those from HF's beam search on a subset of 5k datapoints from the Dolly 15k dataset. I could match 4,879 exactly. I didn't dig into the 121 remaining datapoints as the output was not too dissimilar for them. See [comparison here](https://docs.google.com/spreadsheets/d/1rz4rKW39xJ4zJtDD09NEChBsN_oPivToqE9Zpo8anbo/edit?usp=sharing).
* I added references from the [T5 paper](https://arxiv.org/abs/1910.10683) and other sources where possible in the code documentation. Note though that FLAN-T5 is based on an upgraded version of T5 known as [T5 v.1.1](https://huggingface.co/docs/transformers/model_doc/flan-t5).


**The bad**

(This is actually a TODO list.)
* I have not implemented a training routine yet.
* I have not implemented batched inference in my beam search decoding code.
* I will try other implementations of FlashAttn which may support additive biases.


# Usage

**Dependencies**

`pip install torch numpy transformers`

The Transformers package is used for loading checkpoint weights and the tokenizer.

**Prediction**
```
python run_beam_search.py --model_size large \
	--prompt 'How do I make pesto sauce?' \
	--num_beams 2 \
	--max_new_tokens 100
```

Generates the output:
```
To make pesto sauce, combine 1 cup chopped basil, 1 tablespoon olive oil, 1 teaspoon salt, and 1 teaspoon pepper in a large bowl.
```
