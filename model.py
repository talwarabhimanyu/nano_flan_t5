"""
This is a simple implementation of FLAN-T5 for fun and education. I used
HuggingFace (HF)'s implementation for reference and directly copied
some code snippets (e.g. for LayerNorm without rescaling, computing positional
embeddings) from there. This code wouldn't exist without the two sources
below:

1.  Andrej Karpathy's nanoGPT.
    https://github.com/karpathy/nanoGPT/tree/master
2.  HuggingFace's (HF) PyTorch implementation of T5. 
    https://github.com/huggingface/transformers/tree/main/src/transformers/models/t5

Some key differences:
1. Attention: I've used PyTorch's scaled_dot_product_attention as
   it uses FlashAttention under the hood. That said, to my knowledge,
   flash attn doesn't support additive biases (which is required for
   relative positional embeddings in T5). While I don't see a speed-up
   vs. HF, it seems a matter of time before this support is available.

2. Generate: I added beam search to the generate function. My code makes
   use of KV cache and the inference times were similar to HF in my tests.

3. The code is (hopefully) easier to follow. I've added 
   explanations / references for most code snippets, and 
   often cited excerpts from the T5 paper where helpful. I've
   made some design choices (like keeping position embedding params
   in the EncoderBlock as opposed to the first self attention layer)
   which make the code easy to understand.
"""

from dataclasses import dataclass
import math
import re
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    """
    From the T5 paper:

    <quote>
        We use a simplified version of layer normalization where the 
        activations are only rescaled and no additive bias is applied.
    </quote>

    Following is HF's implementation.
    """
    def __init__(self, ndim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # T5 uses a layer_norm which only scales and doesn't shift, which is also known as Root Mean
        # Square Layer Normalization https://arxiv.org/abs/1910.07467 thus varience is calculated
        # w/o mean and there is no bias. Additionally we want to make sure that the accumulation for
        # half-precision inputs is done in fp32

        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states

class SelfAttention(nn.Module):
    def __init__(self, 
                config,
                is_causal=False):
        super().__init__()
        self.is_causal = is_causal

        # projection of input to queries, keys, values
        self.q_proj = nn.Linear(config.n_embed, config.n_proj, bias=False)
        self.k_proj = nn.Linear(config.n_embed, config.n_proj, bias=False)
        self.v_proj = nn.Linear(config.n_embed, config.n_proj, bias=False)
        
        # projection of output
        self.o_proj = nn.Linear(config.n_proj, config.n_embed, bias=False)
        
        # regularization
        self.dropout_rate = config.dropout_rate
        self.n_heads = config.n_heads
        self.n_embed = config.n_embed
        self.n_proj = config.n_proj

    def forward(self, 
                input,
                return_past_kv=False,
                attn_mask=None,
                pos_embeddings=None,
                past_key_value=None):
        """
        INPUTS:
            (1) input, the input sequence with dimensions (batch_sz x seq_len x n_embed)
            (2) pos_embeddings, (1 x n_heads x seq_len x full_seq_len), even though I
                name the variable 'embeddings', in the case of T5 these are scalars. 
                These scalars are added to unnormalized attention scores. Note that
                full_seq_len includes the length of input tokens as well as those tokens
                that have already been seen so far. For the encoder both of these lengths
                are the same.
            (3) attn_mask, float tensor (1 x 1 x seq_len x full_seq_len). Here, full_seq_len
                refers to length of the to-be-decoded input sequence plus the length of the 
                sequence that we've decoded so far. The attn_mask is used for 'causal'
                attention. Note that this matrix isn't always square.
            (4) past_key_value, tuple of two tensors, each of shape 
                batch_sz x n_heads x past_seq_len x d. One tensor is keys for the sequence 
                decoded so far and one for values. Here past_seq_len = full_seq_len - seq_len.

        OUTPUTS:
            attn_out, tensor (batch_sz x seq_len x n_embed)
            (key, value), OPTIONAL, tuple with the updated key and value for this layer.
        """
        batch_sz, seq_len, n_embed = input.size()
        # project input to query, key, value
        query = self.q_proj(input)                          # batch_sz, seq_len, n_embed
        key = self.k_proj(input)                            # batch_sz, seq_len, n_embed
        value = self.v_proj(input)                          # batch_sz, seq_len, n_embed
        """
        We split n_embed into n_heads chunks of length d each. We 
        transpose because scaled_dot_product_attention requires 
        last two dimensions to be sequence length and embedding size.
        """
        query = query.view(batch_sz, 
                            seq_len, 
                            self.n_heads, 
                            self.n_proj // self.n_heads
                            ).transpose(1, 2) # batch_sz x n_heads x seq_len x d
        key = key.view(batch_sz, 
                            seq_len, 
                            self.n_heads, 
                            self.n_proj // self.n_heads
                            ).transpose(1, 2)   # batch_sz x n_heads x seq_len x d
        value = value.view(batch_sz, 
                            seq_len, 
                            self.n_heads, 
                            self.n_proj // self.n_heads
                            ).transpose(1, 2) # batch_sz x n_heads x seq_len x d
        """
        For efficient decoding, reuse key-value of sequence decoded so far. While
        decoding for token at timestep t, project input[t] to key-value for that one
        time-step, and then concatenate with key[0,1,..t-1] and value[0,1,...t-1].
        Project input[t] to query[t] and have it attend to key-value[0,1,...t-1].
        """
        full_seq_len = seq_len
        if past_key_value is not None:
            key = torch.cat([past_key_value[0], key], dim=2)         # batch_sz x n_heads x full_seq_len x d
            value = torch.cat([past_key_value[1], value], dim=2)     # batch_sz x n_heads x full_seq_len x d
            full_seq_len += past_key_value[0].shape[2]
        
        if not self.is_causal:
            attn_mask = pos_embeddings
        else:
            attn_mask = attn_mask + pos_embeddings          # 1 x n_heads x seq_len x total_seq_len 
        """
        The scaled_dot_product_attention (SDPA) function scales down query-key
        dot products inside SoftMax by division with sqrt(embedding size).
        But the HF implementation doesn't do this scaling. To match our
        output with HF and still use SDPA, I've scaled up query and key by
        an appropriate factor before calling SDPA.
        """
        scale_factor = math.pow(query.shape[-1], 1/4)
        attn_out = F.scaled_dot_product_attention(query=query*scale_factor,
                                            key=key*scale_factor,
                                            value=value,
                                            dropout_p=self.dropout_rate if self.training else 0.,
                                            attn_mask=attn_mask,
                                            is_causal=False)                             # batch_sz x n_heads x seq_len x d
        attn_out = attn_out.transpose(1,2).contiguous().view(batch_sz, seq_len, -1)     # batch_sz x seq_len x n_embed
        attn_out = self.o_proj(attn_out)
        if not return_past_kv:
            return attn_out
        return attn_out, (key, value)

class CrossAttention(nn.Module):
    def __init__(self, 
                config):
        super().__init__()

        # projection of input to queries, and encoder hidden states to keys and values
        self.q_proj = nn.Linear(config.n_embed, config.n_proj, bias=False)
        self.k_proj = nn.Linear(config.n_embed, config.n_proj, bias=False)
        self.v_proj = nn.Linear(config.n_embed, config.n_proj, bias=False)

        # projection of output
        self.o_proj = nn.Linear(config.n_proj, config.n_embed, bias=False)

        # regularization
        self.dropout_rate = config.dropout_rate
        self.n_heads = config.n_heads
        self.n_embed = config.n_embed
        self.n_proj = config.n_proj

    def forward(self, 
            input,
            return_past_kv=False,
            enc_hidden_states=None,
            past_key_value=None):
        """
        INPUTS:
            (1) input: the input sequence, (batch_sz x dec_seq_len x n_embed)
            (2) return_past_kv, boolean, if True return updated key and value
            (3) enc_hidden_states: encoder's last layer hidden states, (batch_sz x enc_seq_len x n_embed)
            (4) past_key_value, tuple of two tensors each of shape batch_sz x n_heads x enc_seq_len x d
                where d = n_embed//n_heads. Note that if past_key_value is provided
                we don't need encoder_hidden_states. Basically past_key_value are projected
                versions of enc_hidden_states, projected using this layer's k_proj and v_proj
                Linear layers.

        OUTPUTS:
            (1) attn_out, tensor (batch_sz x dec_seq_len x n_embed)
            (2) (key, value), OPTIONAL, tuple with the updated key and value for this layer.
        """
        batch_sz, dec_seq_len, n_embed = input.size()
        
        # project input to query, and enc_hidden_states to key and value
        query = self.q_proj(input)                                  # batch_sz x dec_seq_len x n_embed
        query = query.view(batch_sz, 
                        dec_seq_len, 
                        self.n_heads, 
                        self.n_proj // self.n_heads
                        ).transpose(1, 2)               # batch_sz x n_heads x dec_seq_len x n_embed
        if past_key_value is None:
            enc_seq_len = enc_hidden_states.shape[1]
            key = self.k_proj(enc_hidden_states)                        # batch_sz x enc_seq_len x n_embed
            value = self.v_proj(enc_hidden_states)                      # batch_sz x enc_seq_len x n_embed
            key = key.view(batch_sz, 
                            enc_seq_len, 
                            self.n_heads, 
                            self.n_proj // self.n_heads
                            ).transpose(1, 2)               # batch_sz x n_heads x enc_seq_len x n_embed
            value = value.view(batch_sz, 
                            enc_seq_len, 
                            self.n_heads, 
                            self.n_proj // self.n_heads
                            ).transpose(1, 2)               # batch_sz x n_heads x enc_seq_len x n_embed
        else:
            key, value = past_key_value
        """
        See my comment inside the SelfAttention class' forward pass
        for an explanation of this scale_factor.
        """
        scale_factor = math.pow(query.shape[-1], 1/4)
        attn_out = F.scaled_dot_product_attention(query=query*scale_factor,
                                            key=key*scale_factor,
                                            value=value,
                                            dropout_p=self.dropout_rate if self.training else 0.,
                                            is_causal=False)
        attn_out = attn_out.transpose(1,2).contiguous().view(batch_sz, dec_seq_len, -1)     # batch_sz x dec_seq_len x n_embed
        attn_out = self.o_proj(attn_out)
        if not return_past_kv:
            return attn_out
        return attn_out, (key, value)

class FFNGated(nn.Module):
    """
    The FLAN-T5 architecture is based on T5.1.1, an upgrade on the original
    T5. The following webpage describes the main differences vs. T5:

    https://github.com/google-research/text-to-text-transfer-transformer/blob/main/released_checkpoints.md#t511
    
    However, it doesn't describe a key change in the Feed Forward Network used
    inside both encoder and decoder blocks. In T5.1.1., a Gated FFN is used.
    I got to know this from HuggingFace's documentation (see below):

    https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5Config.feed_forward_proj

    The following implementation is taken as it is from HuggingFace's codebase:

    https://github.com/huggingface/transformers/blob/66c240f3c950612fa05b2e14c85d4b86c88e473e/src/transformers/models/t5/modeling_t5.py#L302
    """
    def __init__(self, config):
        super().__init__()
        self.wi_0 = nn.Linear(config.n_embed, config.n_ffn, bias=False)
        self.wi_1 = nn.Linear(config.n_embed, config.n_ffn, bias=False)
        self.wo = nn.Linear(config.n_ffn, config.n_embed, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.gelu = nn.GELU(approximate='tanh')

    def forward(self, input):
        gate_values = self.gelu(self.wi_0(input))
        fc_values = self.wi_1(input)
        hidden_states = fc_values * gate_values
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states

class EncoderBlock(nn.Module):
    """
    From the T5 paper:

    <quote>
        The encoder consists of a stack of “blocks”, each of which 
        comprises two subcomponents: a self-attention layer 
        followed by a small feed-forward network. Layer normalization 
        (Ba et al., 2016) is applied to the input of each subcomponent.

        After layer normalization, a residual skip connection 
        (He et al., 2016) adds each subcomponent’s input to its output.
        Dropout (Srivastava et al., 2014) is applied within the 
        feed-forward network, on the skip connection, on the attention 
        weights, and at the input and output of the entire stack.
    </quote>

    """
    def __init__(self, config):
        super().__init__()
        self.ln_0 = LayerNorm(config.n_embed)
        self.self_attn = SelfAttention(config=config,
                            is_causal=False)
        self.dropout1 = nn.Dropout(config.dropout_rate)
        self.ln_1 = LayerNorm(config.n_embed)
        self.ffn = FFNGated(config=config)
        self.dropout2 = nn.Dropout(config.dropout_rate)
    
    def forward(self, input, pos_embeddings):
        """
        INPUTS:
            (1) input: tensor of shape (batch_sz x enc_seq_len x n_embed)
            (2) pos_embeddings, tensor of shape 1 x n_heads x enc_seq_len x enc_seq_len

        OUTPUTS:
        """
        # Self-attn sub-block
        normed_input = self.ln_0(input)
        attn_out = self.self_attn(normed_input,
                            pos_embeddings=pos_embeddings)
        input = input + self.dropout1(attn_out)

        # FFN sub-block
        normed_input = self.ln_1(input)
        ffn_out = self.ffn(normed_input)
        input = input + self.dropout2(ffn_out)
        return input

class DecoderBlock(nn.Module):
    """
    From the T5 paper:

    <quote>
         The decoder is similar in structure to the encoder except 
         that it includes a standard attention mechanism after each 
         self-attention layer that attends to the output of the encoder.

         The self-attention mechanism in the decoder also uses a form 
         of autoregressive or causal selfattention, which only allows 
         the model to attend to past outputs. The output of the final 
         decoder block is fed into a dense layer with a softmax output, 
         whose weights are shared with the input embedding matrix.
    </quote>

    """

    def __init__(self, config):
        super().__init__()
        self.ln_0 = LayerNorm(config.n_embed)
        self.self_attn = SelfAttention(config=config,
                            is_causal=True)
        self.dropout0 = nn.Dropout(config.dropout_rate)

        self.ln_1 = LayerNorm(config.n_embed)
        self.cross_attn = CrossAttention(config=config)
        self.dropout1 = nn.Dropout(config.dropout_rate)

        self.ln_2 = LayerNorm(config.n_embed)
        self.ffn = FFNGated(config=config)
        self.dropout2 = nn.Dropout(config.dropout_rate)

    def forward(self, 
            input, 
            attn_mask,
            enc_hidden_states, 
            pos_embeddings,
            return_past_kv=False,
            past_key_value=None):
        past_key_value_self, past_key_value_cross = None, None
        if past_key_value is not None:
            past_key_value_self = past_key_value[:2]
            past_key_value_cross = past_key_value[2:]

        # Self-attn sub-block
        normed_input = self.ln_0(input)
        self_attn_out = self.self_attn(input=normed_input,
                            attn_mask=attn_mask,
                            pos_embeddings=pos_embeddings,
                            return_past_kv=return_past_kv,
                            past_key_value=past_key_value_self)
        if isinstance(self_attn_out, tuple):
            attn_out, past_key_value_self = self_attn_out
        else:
            attn_out = self_attn_out
        input = input + self.dropout0(attn_out)

        # Cross-attn sub-block
        normed_input = self.ln_1(input)
        cross_attn_out = self.cross_attn(input=normed_input,
                                enc_hidden_states=enc_hidden_states,
                                return_past_kv=return_past_kv,
                                past_key_value=past_key_value_cross)
        if isinstance(cross_attn_out, tuple):
            attn_out, past_key_value_cross = cross_attn_out
        else:
            attn_out = cross_attn_out

        input = input + self.dropout1(attn_out)
        
        # FFN sub-block
        normed_input = self.ln_2(input)
        ffn_out = self.ffn(normed_input)
        input = input + self.dropout2(ffn_out)

        if not return_past_kv:
            return input
        return input, past_key_value_self + past_key_value_cross

@dataclass
class FLANT5Config:
    vocab_size: int = 32128
    n_layers: int = 8
    n_heads: int = 6
    n_embed: int = 512
    n_proj: int = 384
    n_ffn: int = 1024
    dropout_rate: float = 0.1
    relative_attention_num_buckets: int = 32
    relative_attention_max_distance: int = 128

class FLANT5(nn.Module):
    def __init__(self, config, device=None):
        super().__init__()
        self.config = config
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance
        self.enc_dec_embeddings = nn.Embedding(config.vocab_size, config.n_embed)

        # Encoder parameters
        self.enc_pos_embeddings = nn.Embedding(config.relative_attention_num_buckets, 
                                                        config.n_heads)
        self.enc_in_dropout = nn.Dropout(config.dropout_rate)
        self.enc_final_ln = LayerNorm(config.n_embed)
        self.enc_final_dropout = nn.Dropout(config.dropout_rate)
        self.encoder = nn.ModuleList(
                        [EncoderBlock(config=config) for i in range(config.n_layers)]
                        )

        # Decoder parameters
        self.dec_pos_embeddings = nn.Embedding(config.relative_attention_num_buckets, 
                                                        config.n_heads)
        self.dec_in_dropout = nn.Dropout(config.dropout_rate)
        self.dec_final_ln = LayerNorm(config.n_embed)
        self.dec_final_dropout = nn.Dropout(config.dropout_rate)
        self.decoder = nn.ModuleList(
                        [DecoderBlock(config=config) for i in range(config.n_layers)]
                        )
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)

    def forward(self, 
            enc_input_ids, 
            dec_input_ids,
            return_past_kv=False,
            enc_last_hidden_states=None,
            past_key_value=None,
            ):
        """
        INPUTS
            1. enc_input_ids
            2. dec_input_ids
            3. enc_last_hidden_states [OPTIONAL], tensor of shape 
               batch_sz x enc_seq_len x n_embed
            4. past_key_value [OPTIONAL], tuple of length n_layers where
               ith element is a tuple with 4 tensors. The firs two are 
               keys-values for past tokens in the self-attention block
               of decoder layer i (each of shape 
               batch_sz x n_heads x dec_seq_len-1 x n_embed_per_head.) 
               The last two are keys-values for the cross-attention block. 
               Each of shape batch_sz x n_heads x enc_seq_len x n_embed_per_head
        """
        # Encoder stack forward pass
        if enc_last_hidden_states is None and (past_key_value is None or len(past_key_value) == 2):
            enc_input_embeds = self.enc_dec_embeddings(enc_input_ids)
            enc_hidden_states = self.enc_in_dropout(enc_input_embeds)
            enc_pos_embeddings = self.compute_pos_embeddings(
                                                    query_len=enc_hidden_states.shape[1],
                                                    key_len=enc_hidden_states.shape[1],
                                                    device=enc_input_ids.device,
                                                    is_encoder=True)
            enc_pos_embeddings.to(self.device);
            for i, layer_module in enumerate(self.encoder):
                layer_outputs = layer_module(input=enc_hidden_states,
                                            pos_embeddings=enc_pos_embeddings)
                enc_hidden_states = layer_outputs
            enc_last_hidden_states = self.enc_final_ln(enc_hidden_states)
            enc_last_hidden_states = self.enc_final_dropout(enc_last_hidden_states)
        
        # Decoder stack forward pass
        dec_input_embeds = self.enc_dec_embeddings(dec_input_ids)   # batch_sz x dec_seq_len x n_embed
        dec_hidden_states = self.dec_in_dropout(dec_input_embeds)   # batch_sz x dec_seq_len x n_embed
        dec_seq_len = dec_hidden_states.shape[1]
        if past_key_value is not None:
            # add length of sequence decoded so far
            total_dec_seq_len = dec_seq_len + past_key_value[0][0].shape[2]
        else:
            total_dec_seq_len = dec_seq_len
        """
        Create position embeddings according to total_dec_seq_len,
        although we'll only need to use the entries corresopnding to
        the sequence that has not been decoded so far.
        """
        dec_pos_embeddings = self.compute_pos_embeddings(
                                            query_len=total_dec_seq_len,
                                            key_len=total_dec_seq_len,
                                            device=dec_input_ids.device,
                                            is_encoder=False)   # 1 x n_heads x total_dec_seq_len x total_dec_seq_len
        dec_pos_embeddings = dec_pos_embeddings.to(self.device)
        """
        Create a causal mask for decoder self-attention. This will be the
        same for all decoder layers.
        """
        attn_mask = torch.masked_fill(torch.zeros(dec_seq_len, total_dec_seq_len),
                                        torch.tril(torch.ones(dec_seq_len, total_dec_seq_len),
                                                    diagonal=total_dec_seq_len-dec_seq_len
                                                    ) == 0,
                                        float('-inf')
                                    ).view(1, 1, 
                                            dec_seq_len, 
                                            total_dec_seq_len)    # 1 x 1 x dec_seq_len x total_dec_seq_len
        attn_mask = attn_mask.to(self.device)

        updated_past_key_value = ()
        for i, layer_module in enumerate(self.decoder):
            layer_outputs = layer_module(input=dec_hidden_states,
                                        attn_mask=attn_mask,
                                        enc_hidden_states=enc_last_hidden_states,
                                        pos_embeddings=dec_pos_embeddings[:,:,-dec_seq_len:,:],
                                        return_past_kv=return_past_kv,
                                        past_key_value=None if past_key_value is None
                                                            else past_key_value[i]
                            )
            if isinstance(layer_outputs, tuple):
                dec_hidden_states, layer_kv = layer_outputs
                updated_past_key_value = updated_past_key_value + (layer_kv,)
            else:
                dec_hidden_states = layer_outputs
        dec_hidden_states = self.dec_final_ln(dec_hidden_states)
        dec_hidden_states = self.dec_final_dropout(dec_hidden_states)

        # Compute output logits
        lm_logits = self.lm_head(dec_hidden_states)     # batch_sz x dec_seq_len x vocab_size
        if not return_past_kv:
            return lm_logits
        return lm_logits, updated_past_key_value

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        From the T5 paper:

        <quote>
            We use a simplified form of position embeddings where each “embedding” 
            is simply a scalar that is added to the corresponding logit used 
            for computing the attention weights. For efficiency, we also share 
            the position embedding parameters across all layers in our model, 
            though within a given layer each attention head uses a different 
            learned position embedding.

            Typically, a fixed number of embeddings are learned, each corresponding 
            to a range of possible key-query offsets. In this work, we use 32 
            embeddings for all of our models with ranges that increase in size 
            logarithmically up to an offset of 128 beyond which we assign all 
            relative positions to the same embedding.
        </quote>

        The following implementation is taken from HF codebase, which in turn cites
        Mesh Tensorflow as the source:
        https://github.com/huggingface/transformers/blob/e42587f596181396e1c4b63660abf0c736b10dae/src/transformers/models/t5/modeling_t5.py#L389
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            """
            If bidirectional, we treat relative position between (query_i, key_j)
            and (query_j, key_i) as the same.
            """
            relative_position = torch.abs(relative_position)
        else:
            """
            If not bidirectional, we set all relative positions where a query is 
            'looking into the future keys' to 0. We use the absolute value of all
            negative relative positions.
            """
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_position_if_large = max_exact + (
                                        torch.log(relative_position.float() / max_exact)
                                        / math.log(max_distance / max_exact)
                                        * (num_buckets - max_exact)
                                    ).to(torch.long)
        relative_position_if_large = torch.min(
                                        relative_position_if_large, 
                                        torch.full_like(relative_position_if_large, 
                                                        num_buckets - 1)
                                        )

        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets

    def compute_pos_embeddings(self, query_len, key_len, device, is_encoder=True):
        """
        The following implementation is taken from HF codebase:
        https://github.com/huggingface/transformers/blob/e42587f596181396e1c4b63660abf0c736b10dae/src/transformers/models/t5/modeling_t5.py#L436
        """
        query_indices = torch.arange(query_len, 
                                    dtype=torch.long, 
                                    device=device)[:, None]
        key_indices = torch.arange(key_len, 
                                    dtype=torch.long, 
                                    device=device)
        relative_position = key_indices - query_indices     # Shape query_len x  key_len
        relative_position_bucket = self._relative_position_bucket(
                                        relative_position,
                                        bidirectional=is_encoder,
                                        num_buckets=self.relative_attention_num_buckets,
                                        max_distance=self.relative_attention_max_distance,
                                        )
        if is_encoder:
            values = self.enc_pos_embeddings(relative_position_bucket)     # query_len x key_len x num_heads
        else:
            values = self.dec_pos_embeddings(relative_position_bucket)    # query_len x key_len x num_heads
        values = values.permute([2, 0, 1]).unsqueeze(0)                         # 1 x num_heads x query_len x key_len
        return values

    @classmethod
    def from_pretrained(cls, model_type):
        from transformers import T5ForConditionalGeneration
        
        # Hugginface model
        hf_model = T5ForConditionalGeneration.from_pretrained(model_type)
        hf_sd = hf_model.state_dict()

        # Available configs
        config_args = {
                "google/flan-t5-small" : dict(n_layers=8, n_heads=6, n_embed=512, n_proj=384, n_ffn=1024, vocab_size=32128),
                "google/flan-t5-base" : dict(n_layers=12, n_heads=12, n_embed=768, n_proj=768, n_ffn=2048, vocab_size=32128),
                "google/flan-t5-large" : dict(n_layers=24, n_heads=16, n_embed=1024, n_proj=1024, n_ffn=2816, vocab_size=32128)
                }[model_type]

        # Our model
        config = FLANT5Config(**config_args)
        model = FLANT5(config=config)
        sd = model.state_dict()

        # Map HF model state dict key names to match our state dict
        hf_to_our_keys = {}
        ignore_list = ["embed_tokens"]
        for k in hf_sd.keys():
            if any([v in k for v in ignore_list]):
                continue
            new_k = k
            replace_dict = {
                    "encoder.block.0.layer.0.SelfAttention.relative_attention_bias" : "enc_pos_embeddings",
                    "decoder.block.0.layer.0.SelfAttention.relative_attention_bias" : "dec_pos_embeddings",
                    "SelfAttention" : "self_attn",
                    "EncDecAttention" : "cross_attn",
                    "DenseReluDense" : "ffn",
                    "shared" : "enc_dec_embeddings",
                    "encoder.final_layer_norm" : "enc_final_ln",
                    "decoder.final_layer_norm" : "dec_final_ln",
                    }
            for find_str, replace_str in replace_dict.items():
                new_k = new_k.replace(find_str, replace_str)
            new_k = re.sub(r"(layer\.)([0-9]+)(\.layer_norm)", r"\1\2.ln_\2", new_k)
            new_k = re.sub(r"\.block(\.[0-9]+)\.layer\.[0-9]+", r"\1", new_k)
            new_k = re.sub(r"(\.(q|k|v|o))\.", r"\1_proj.", new_k)

            # Store new key name
            hf_to_our_keys[k] = new_k
        
        # Copy parameter values
        for k in hf_sd.keys():
            if any([v in k for v in ignore_list]):
                continue
            with torch.no_grad():
                sd[hf_to_our_keys[k]].copy_(hf_sd[k])
        return model

    @torch.no_grad()
    def generate(self, 
                prompt_text, 
                tokenizer, 
                max_new_tokens, 
                num_beams=1):
        """
        INPUTS
            (1) prompt_text, str
            (2) tokenizer, this is the HuggingFace tokenizer object
                initialized for flan-t5
            (3) max_new_tokens, int
            (4) num_beam, int >= 1
        """
        # For FLANT5, the first decoder input is the pad token.
        bos_token_id, eos_token_id = tokenizer.pad_token_id, tokenizer.eos_token_id
        prompt_input_ids = tokenizer(prompt_text, 
                                return_tensors="pt"
                                ).input_ids
        enc_input_ids = prompt_input_ids.to(self.device)    # 1 x enc_seq_len
        dec_input_ids = torch.tensor([[bos_token_id]],
                                    device=self.device,
                                )  # 1 x 1
        """
        Prepare the first set of num_beams beams for decoder. We feed
        the bos_token_id to the decoder, and then take the top num_beams
        output token ids. These token ids comprise the first beam.
        """
        logits, past_key_value = self(enc_input_ids=enc_input_ids,
                                    dec_input_ids=dec_input_ids,
                                    return_past_kv=True,
                                    past_key_value=None,
                                    )    # logits: 1 x 1 x vocab_size
        vocab_size = logits.shape[2]
        log_probs = torch.log_softmax(logits[0,0,:], dim=0)    # vocab_size
        top_log_probs, top_token_indices = log_probs.topk(2*num_beams)
        is_beam_done = top_token_indices == eos_token_id
        dec_input_ids = top_token_indices[~is_beam_done][:num_beams].unsqueeze(1)   # num_beams x 1
        running_log_probs = top_log_probs[~is_beam_done][:num_beams].unsqueeze(1)   # num_beams x 1
        updated_kv_cache = []
        for layer in past_key_value:
            updated_kv_cache.append(tuple([kv.repeat(num_beams, 1, 1, 1) for kv in layer]))
        past_key_value = tuple(updated_kv_cache)
        output_token_ids = [[tok_id.item()] for tok_id in dec_input_ids[:,0]]

        """
        Start beam search decoding.
        """
        concluded_beams, concluded_scores = [], []
        concluded_beams_min_score = 0.
        for i in range(max_new_tokens-1):
            logits, past_key_value = self(enc_input_ids=enc_input_ids,
                                    dec_input_ids=dec_input_ids,
                                    return_past_kv=True,
                                    past_key_value=past_key_value,
                                    )    # logits: num_beams x 1 x vocab_size
            log_probs = torch.log_softmax(logits[:,0,:], 
                                    dim=1)   # num_beams x vocab_size

            """
            running_log_prob is of shape (num_beams x 1) and the ith index
            has the log-prob for the ith beam. Adding log_probs to
            running_log_probs gives a tensor of shape (num_beams x vocab_size).
            The [i,j]the entry gives the log-prob of the ith beam, if we were
            to append token j (where j=0,..,vocab_size-1) to the ith beam's
            token sequence.
            """
            updated_log_probs = running_log_probs + log_probs
            """
            Each of vocab_size tokens could be appended to each of the num_beams
            beams, and so now we have num_beams*vocab_size candidate beams to
            select from. We flatten updated_log_probs to get a tensor of size
            num_beams*vocab_size.

            We then select the top 2*num_beams candidates. We select 2x because
            if some of the top num_beams candidates were eos, they would not
            be pursued further, and so we would be left with less than num_beams
            beams for further processing.
            """
            top_running_log_probs, top_indices = updated_log_probs.flatten(
                                                    ).topk(2*num_beams)
            """
            (1) top_beams_index gives the index of beams in [0,...num_beams-1] which
            survived the addition of this new token.
            (2) top_tokens_vocab_index gives the index in vocabulary (0,...vocab_size-1)
            of the tokens appended to beams in top_beams_index
            """
            top_tokens_vocab_index = top_indices % vocab_size
            top_beams_index = top_indices // vocab_size
            """
            Some beams would've reached the eos token. We will not consider these
            beams for further generation.
            """
            indices_to_pursue = []
            new_concluded_beam_scores = []
            for j, (token_vocab_idx, beam_idx) in enumerate(zip(
                                                    top_tokens_vocab_index,
                                                    top_beams_index)):
                if token_vocab_idx == eos_token_id:
                    concluded_beams.append(tokenizer.decode(output_token_ids[beam_idx],
                                                skip_special_tokens=True))
                    new_concluded_beam_scores.append(top_running_log_probs[j].item())
                else:
                    indices_to_pursue.append(j)
                if len(indices_to_pursue) == num_beams:
                    break
            
            if new_concluded_beam_scores:
                concluded_scores.extend(new_concluded_beam_scores)
                concluded_beams_min_score = min(concluded_beams_min_score,
                                                new_concluded_beam_scores[-1])

            running_log_probs = top_running_log_probs[indices_to_pursue].unsqueeze(1)       # num_beams x 1
            selected_beams = top_beams_index[indices_to_pursue]                     # num_beams
            """
            Exit condition: If the maximum running score of active beams is less 
            than the minimum score of concluded beams 
            
            AND 

            We have num_beams concluded beams.
            """
            if len(concluded_beams) >= 1 and running_log_probs[0,0] <= concluded_beams_min_score:
                break

            dec_input_ids = top_tokens_vocab_index[indices_to_pursue].unsqueeze(1) # num_beams x 1
            if output_token_ids is None:
                new_output_token_ids = [[tok_id.item()] for tok_id in dec_input_ids[:,0]]
            else:
                new_output_token_ids = [output_token_ids[bidx] + [tok_id.item()] 
                                            for bidx, tok_id in 
                                            zip(selected_beams, dec_input_ids[:,0])]
            output_token_ids = new_output_token_ids
            """
            We keep the kv cache only for this beams which survived.
            """
            selected_past_key_value = []
            for layer in past_key_value:
                kv_cross_attn = layer[2:]
                kv_self_attn = tuple(kv[selected_beams, :, :, :] 
                                        for kv in layer[:2])        # tuple of two tensor(num_beams x n_heads x len x n_ebed)
                selected_past_key_value.append(kv_self_attn + kv_cross_attn)
            past_key_value = tuple(selected_past_key_value)
        if len(concluded_beams) < num_beams: 
            concluded_beams.extend(tokenizer.batch_decode([x for x in output_token_ids],
                                        skip_special_tokens=True))
            concluded_scores.extend(running_log_probs[:,0].cpu().tolist())

        sorted_idx = np.argsort(concluded_scores)[::-1]
        sorted_sents = [concluded_beams[sidx].strip() 
                        for sidx in sorted_idx]
        return sorted_sents[0]
