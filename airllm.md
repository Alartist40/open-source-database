# AirLLM: Scaling Large Language Models on Low-End Commodity Computers

## 1. Introduction
AirLLM is an open-source library designed to optimize the inference of Large Language Models (LLMs), enabling massive models like Llama 3 (70B or even 405B) to run on consumer-grade hardware with as little as 4GB-8GB of VRAM. It achieves this without requiring traditional model compression techniques like distillation or pruning, though it does support quantization for additional performance.

The core innovation of AirLLM is its **layer-wise inference** approach, which drastically reduces the peak memory requirement by only keeping a single layer of the model in GPU memory at any given time.

---

## 2. Technical Specifications
- **Primary Language**: Python 3.x
- **Core Frameworks**: PyTorch, Hugging Face Transformers, Accelerate.
- **Hardware Backends**:
    - NVIDIA GPU (via CUDA)
    - Apple Silicon (via MLX)
    - CPU (Fallback)
- **Model Support**: LLaMA 2/3/3.1, Qwen 1.5/2, Baichuan 2, ChatGLM 3, InternLM, Mistral, Mixtral (MoE).

---

## 3. Architecture Overview
AirLLM follows a modular design to decouple architecture-specific logic from the core execution engine.

### Core Modules:
- **`AutoModel`**: Acts as a factory class. It inspects the `config.json` of a model to determine the correct `AirLLM` subclass to instantiate.
- **`AirLLMBaseModel`**: The foundational class containing the layer-wise execution loop, memory management, and quantization logic.
- **`ModelPersister`**: An abstraction for the storage layer, allowing different weight formats (Safetensors vs. MLX NPZ).
- **`utils.py`**: A utility hub for model sharding, disk space checking, and memory cleanup.

---

## 4. Directory & File Structure
Detailed listing of the core `air_llm` package:

```text
air_llm/
└── airllm/
    ├── __init__.py           # Package exports
    ├── airllm_base.py        # Core Engine: Layer-wise inference logic
    ├── airllm.py             # Llama2/Llama3 implementation (inherits Base)
    ├── auto_model.py         # Model dispatcher/factory
    ├── airllm_baichuan.py    # Baichuan 2 support
    ├── airllm_chatglm.py     # ChatGLM 3 support
    ├── airllm_internlm.py    # InternLM support
    ├── airllm_mistral.py     # Mistral 7B support
    ├── airllm_mixtral.py     # Mixtral 8x7B (MoE) support
    ├── airllm_qwen.py        # Qwen support
    ├── airllm_qwen2.py       # Qwen 2 / 2.5 support
    ├── airllm_llama_mlx.py   # Native MacOS (MLX) support
    ├── profiler.py           # LayeredProfiler for performance tracking
    ├── utils.py              # Sharding, Memory Cleanup, Quantization
    ├── tokenization_baichuan.py # Custom tokenizer for Baichuan
    └── persist/              # Persistence Layer
        ├── __init__.py
        ├── model_persister.py # Abstract base class for persistence
        ├── safetensor_model_persister.py # CUDA/PyTorch backend
        └── mlx_model_persister.py        # Apple Silicon backend
```

---

## 5. The Inference Engine (`AirLLMBaseModel`)
The engine's primary job is to orchestrate the movement of weights from disk -> CPU RAM -> GPU VRAM -> Purge.

### 5.1 The `forward` Method
The `forward` pass is where the layer-wise magic happens. Instead of a single call to a model object, AirLLM iterates through a list of layers.

```python
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if cache_utils_installed:
            # we don't support kv cache for new version yet
            use_cache = False

        if self.profiling_mode:
            self.profiler.clear_profiling_time()
            forward_start = time.process_time()
            forward_start_wall = time.time()

        # Reboot the model to make sure buffers are loaded and memory is clean
        del self.model
        clean_memory()
        self.init_model()

        batch = [input_ids_unit.to(self.running_device).unsqueeze(0) for input_ids_unit in input_ids]
        n_seq = len(batch[0])

        # Create attention mask for the largest input, and position ids to use KV cache
        attention_mask = torch.ones(self.max_seq_len, self.max_seq_len)
        attention_mask = attention_mask.triu(diagonal=1)[None, None, ...] == 0
        attention_mask = attention_mask.to(self.running_device)
        position_ids = torch.arange(self.max_seq_len, dtype=torch.long, device=self.running_device)[None, :]

        kv_cache_list = [] if use_cache else None
        if use_cache:
            for x in self.layers:
                kv_cache_list.append(([], []))
        all_hidden_states = [] * len(self.layers) if output_hidden_states else None
        all_self_attns = [] * len(self.layers) if output_attentions else None

        with torch.inference_mode(), ThreadPoolExecutor() as executor:

            # Load first layer
            if self.prefetching:
                future = executor.submit(self.load_layer_to_cpu, self.layer_names[0])

            for i, (layer_name, layer) in tqdm(enumerate(zip(self.layer_names, self.layers)),
                                               desc=f'running layers({self.running_device})',
                                               total=len(self.layers)):

                if self.prefetching:
                    # Load current layer and prepare next layer
                    state_dict = future.result()
                    moved_layers = self.move_layer_to_device(state_dict)

                    # kick off next layer loading
                    if (i + 1) < len(self.layer_names):
                        future = executor.submit(self.load_layer_to_cpu, self.layer_names[i+1])
                else:
                    state_dict = self.load_layer_to_cpu(layer_name)
                    moved_layers = self.move_layer_to_device(state_dict)

                # Run layer
                for j, seq in enumerate(batch):
                    if layer_name == self.layer_names_dict['embed']:
                        batch[j] = layer(seq)
                    elif layer_name == self.layer_names_dict['norm']:
                        batch[j] = self.run_norm(layer, seq)
                    elif layer_name == self.layer_names_dict['lm_head']:
                        batch[j] = self.run_lm_head(layer, seq)
                    else:
                        # Transformer Layer logic
                        len_seq = self.get_sequence_len(seq)
                        pos_embed_args = self.get_pos_emb_args(0, len_seq)
                        attention_mask_args = self.get_attention_mask_args(attention_mask, 0, len_seq)
                        position_ids_args = self.get_position_ids_args(position_ids, 0, len_seq)

                        kwargs = {'use_cache': use_cache,
                                  'attention_mask': attention_mask[:, :, -len_seq:, -len_seq:],
                                  }
                        kwargs = {**kwargs, **pos_embed_args, **attention_mask_args, **position_ids_args}
                        layer_out = layer(seq, **kwargs)

                        if not use_cache:
                            new_seq = layer_out[0]
                        else:
                            new_seq, (k_cache, v_cache) = layer_out
                            kv_cache_list[i][0].append(k_cache)
                            kv_cache_list[i][1].append(v_cache)
                        batch[j] = new_seq

                # Remove previous layer from memory
                if self.hf_quantizer is not None:
                    for param_name in moved_layers:
                        set_module_tensor_to_device(self.model, param_name, 'meta')
                else:
                    layer.to("meta")
                clean_memory()

        logits = torch.cat(batch, 0)
        return CausalLMOutputWithPast(logits=logits, ...)
```

### 5.2 Memory Management Logic
AirLLM relies on `clean_memory()` to prevent VRAM and RAM fragmentation.

```python
def clean_memory():
    import gc, ctypes, torch
    gc.collect() # Standard Python cleanup
    try:
        # CRITICAL: Force the C runtime to release freed blocks to the OS
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except: pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache() # Release PyTorch's internal GPU cache
```

---

## 6. Sharding & Splitting Logic
Since standard models are too large to load even into CPU RAM in one piece, AirLLM converts them into a one-file-per-layer format.

### `split_and_save_layers` Implementation Detail:
```python
    for layer in tqdm(layers):
        # 1. Identify which original shard contains this layer's weights
        shards = [int(v.split('-')[1]) for k, v in index.items() if k.startswith(layer) and '-' in v]

        # 2. Load the necessary shard(s)
        if len(shards) > 0:
            if max(shards) > shard:
                shard += 1
                to_load = checkpoint_path / f'model-000{shard:02d}-of-000{n_shards:02d}.safetensors'
                state_dict.update(load_file(to_load, device='cpu'))
        else:
            # Single file model
            single_modelfile = [v for k, v in index.items() if k.startswith(layer)][0]
            state_dict.update(load_file(checkpoint_path / single_modelfile, device='cpu'))

        # 3. Extract weights for just THIS layer
        layer_state_dict = dict([(k, v) for k, v in state_dict.items() if k.startswith(layer)])

        # 4. Apply compression
        layer_state_dict = compress_layer_state_dict(layer_state_dict, compression)

        # 5. Save layer weights to disk
        ModelPersister.get_model_persister().persist_model(layer_state_dict, layer, saving_path)

        # 6. Immediate memory purge from CPU dictionary
        for k in layer_state_dict.keys():
            if k in state_dict: del state_dict[k]
        del layer_state_dict
        clean_memory()
```

---

## 7. Model Compression (Quantization)
AirLLM implements Weight-Only quantization using the `bitsandbytes` (bnb) library.

### 7.1 Compression Logic
```python
def compress_layer_state_dict(layer_state_dict, compression=None):
    if compression == '4bit':
        compressed_layer_state_dict = {}
        for k, v in layer_state_dict.items():
            v_quant, quant_state = bnb.functional.quantize_nf4(v.cuda(), blocksize=64)
            compressed_layer_state_dict[k] = v_quant
            for qs_k, qs_v in save_quant_state_to_dict(quant_state).items():
                compressed_layer_state_dict[k + ".4bit." + qs_k] = qs_v
        return compressed_layer_state_dict
    elif compression == '8bit':
        # blocksize 2048 for 8bit blockwise quantization
        # ...
```

---

## 8. Hardware Backends
### 8.1 CUDA (NVIDIA)
The primary backend. Features:
- **Prefetching**: Uses a background thread and `concurrent.futures.ThreadPoolExecutor` to load the next layer from disk, overlapping I/O with GPU computation.
- **Pinned Memory**: Uses `tensor.pin_memory()` to accelerate CPU-to-GPU transfers.

### 8.2 MacOS (MLX)
Implemented in `AirLLMLlamaMlx`.
- **Framework**: Uses Apple's MLX for native Metal performance.
- **Lazy Evaluation**: Uses `mx.eval()` to force the execution of the computational graph at the end of each layer.

---

## 9. Architecture Specifics
AirLLM handles architecture differences by overriding base methods. Example for **QWen**:
```python
class AirLLMQWen(AirLLMBaseModel):
    def set_layer_names_dict(self):
        self.layer_names_dict = {
            'embed': 'transformer.wte',
            'layer_prefix': 'transformer.h',
            'norm': 'transformer.ln_f',
            'lm_head': 'lm_head',
        }

    def get_pos_emb_args(self, len_p, len_s):
        # Specific Rotary positional embedding logic for Qwen
        # ...
```

---

## 10. Installation, Usage & Troubleshooting

### 10.1 Installation
```bash
pip install airllm
# Optional for quantization:
pip install bitsandbytes
```

### 10.2 Usage Example
```python
from airllm import AutoModel

# Load 70B model on 8GB GPU
model = AutoModel.from_pretrained("meta-llama/Llama-2-70b-hf", compression='4bit')

input_tokens = model.tokenizer(["Hello, my name is"], return_tensors="pt")
output = model.generate(input_tokens['input_ids'].cuda(), max_new_tokens=20)
print(model.tokenizer.decode(output[0]))
```

### 10.3 Troubleshooting
- **MetadataIncompleteBuffer Error**: Usually caused by running out of disk space during the splitting process. Ensure enough disk space and clear the HF cache.
- **Out of Memory (OOM)**: Try setting `compression='4bit'` or reducing `max_seq_len`.
- **401 Client Error**: Some models (like Llama 2/3) are gated. Provide your Hugging Face token: `AutoModel.from_pretrained(..., hf_token='YOUR_TOKEN')`.

---
*Created by Jules, AI Software Engineer.*
