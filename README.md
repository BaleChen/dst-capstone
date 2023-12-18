# Dialogue State Tracking with Large Language Models

This is the NYU Shanghai Undergraduate Computer Science and Data Science Capstone Research Project by Bale Chen, Peiyang Wu, and Xiaocheng Yang, supervised by Prof. Wilson Tam at NYU Shanghai. Our main contributions are:
- We proved the feasibility of using LLMs for dialogue state tracking (DST) and achieved the new state of the art on the MultiWOZ 2.2 and 2.4 benchmarks.
- We demonstrated that the slot-level question-answering formulation enhanced LLM’s performance on DST.
- We clarified the influence of model scaling and data scaling on DST performance.

## A Note on FP16 Training with Llama-2

It has been a known issue that llama-2 training with FP16 precision (i.e. mixed-precision training) causes nan values in the hidden states and hence 0 loss values. See [this issue](https://github.com/huggingface/transformers/issues/25065) for more information. The detailed cause is explained in [Bale Chen's comment](https://github.com/huggingface/transformers/issues/25065#issuecomment-1806795798).

To work around this in our project, we first build ``transformers`` from source
```
git clone https://github.com/huggingface/transformers.git
cd transformers
pip install -e .
```

In ``src/transformers/models/llama/modeling_llama.py``, we go to the definition of the forward function of ``LlamaDecoderLayer``. Since the issue is caused by the combination of the inf values from ``mlp`` and the computation inside ``input_layernorm``. We add the following lines before ``hidden_states = self.input_layernorm(hidden_states)`` is called. 

```
# HACK fix llama2's nan value issue when fp16 precision is used
max_dtype = torch.finfo(hidden_states.dtype).max
clamp_value = torch.where(torch.isinf(hidden_states).any(), max_dtype * 0.99, max_dtype)
hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
```
The above codeblock would clamp the inf values in hidden_states to a very large number but does not overflow, which would patch up the numerical instabilities. 

Note that this method is only a workaround as we don't have immediate access to Ampere GPUs that supports bf16 training. Llama-2 series are originally trained with bf16 precision that have the full range of float32, so it is more natural to use bf16 if you have a compatible GPU. Or, if you have time, fp32 training is always fine.

Update: This phenomenon is only observed in Llama-2-7b. The 13b version works fine with fp16 precision.