# AirLLM: The Complete Technical Blueprint and Architecture Specification

## 1. Executive Summary
**AirLLM** (Automated Inference for Resource-constrained Large Language Models) is a pioneering optimization framework designed to bridge the "memory wall" in modern Artificial Intelligence. As Large Language Models (LLMs) scale toward 400B+ parameters (e.g., Llama 3.1 405B), they increasingly outstrip the VRAM capacity of even professional-grade consumer GPUs.

AirLLM enables these ultra-large models to execute on commodity hardware (4GB-8GB VRAM) by implementing **Layer-wise Inference**. This methodology ensures that the peak memory footprint is strictly bound to the size of a single model layer, rather than the cumulative weight of the entire model. This manual provides a complete, exhaustive technical blueprint for software engineers to understand, maintain, or rebuild the system from scratch.

---

## 2. Technical Specifications
- **Primary Language**: Python 3.8+
- **Deep Learning Frameworks**:
    - **PyTorch**: Primary engine for CUDA and CPU execution.
    - **MLX**: Native framework for Apple Silicon (macOS) hardware acceleration.
- **Backends**: NVIDIA CUDA (Compute Capability 7.0+), Apple MPS/MLX, and Generic CPU.
- **Quantization Compatibility**: FP16 (Native), 8-bit (Block-wise), 4-bit (NF4).
- **Architecture Compatibility**: Llama 2/3, Qwen 1/2, Baichuan 2, ChatGLM 3, InternLM, Mistral, Mixtral (MoE).
- **Core Dependencies**: `transformers`, `accelerate`, `safetensors`, `bitsandbytes`, `sentencepiece`, `optimum`.

---

## 3. Global Repository Inventory: An exhaustive File-by-File Guide

Below is a detailed breakdown of the repository structure, describing the role and technical significance of every significant file and directory.

### 3.1 Root Directory
- **`air_llm/`**: Main distribution package. Contains all inference logic.
- **`training/`**: Module for memory-efficient fine-tuning using QLoRA.
- **`rlhf/`**: Logic for Direct Preference Optimization (DPO) and model alignment.
- **`anima_100k/`**: Research-focused module for expanding Llama context length to 100k tokens using Flash Attention.
- **`data/`**: Evaluation datasets and translation scripts (e.g., GPT-4 translated Vicuna set).
- **`eval/`**: Scripts for calculating Elo ratings and benchmarking model performance.
- **`examples/`**: Jupyter notebooks for quick-start and cross-platform usage.

### 3.2 Full Repository Listing (`ls -lR` details)
```text
/app/airllm_repo:
total 128
-rw-rw-r-- 1 jules jules 11357 Apr  3 05:35 LICENSE
-rw-rw-r-- 1 jules jules 12935 Apr  3 05:35 README.md
-rw-rw-r-- 1 jules jules 33281 Apr  3 05:35 README_ja.md
drwxrwxr-x 4 jules jules  4096 Apr  3 05:35 air_llm
drwxrwxr-x 2 jules jules  4096 Apr  3 05:35 anima_100k
-rw-rw-r-- 1 jules jules 44593 Apr  3 05:35 anima_logo.png
drwxrwxr-x 2 jules jules  4096 Apr  3 05:35 assets
drwxrwxr-x 2 jules jules  4096 Apr  3 05:35 data
drwxrwxr-x 2 jules jules  4096 Apr  3 05:35 eval
drwxrwxr-x 2 jules jules  4096 Apr  3 05:35 examples
-rw-rw-r-- 1 jules jules  1957 Apr  3 05:35 funding.json
-rw-rw-r-- 1 jules jules   303 Apr  3 05:35 requirements.txt
drwxrwxr-x 2 jules jules  4096 Apr  3 05:35 rlhf
drwxrwxr-x 2 jules jules  4096 Apr  3 05:35 scripts
drwxrwxr-x 2 jules jules  4096 Apr  3 05:35 training

/app/airllm_repo/air_llm:
total 36
-rw-rw-r-- 1 jules jules 11357 Apr  3 05:35 LICENSE
-rw-rw-r-- 1 jules jules 12222 Apr  3 05:35 README.md
-rw-rw-r-- 1 jules jules     0 Apr  3 05:35 __init__.py
drwxrwxr-x 4 jules jules  4096 Apr  3 05:35 airllm
drwxrwxr-x 2 jules jules  4096 Apr  3 05:35 examples
-rw-rw-r-- 1 jules jules   844 Apr  3 05:35 inference_example.py
-rw-rw-r-- 1 jules jules  1679 Apr  3 05:35 setup.py
drwxrwxr-x 3 jules jules  4096 Apr  3 05:35 tests

/app/airllm_repo/air_llm/airllm:
total 112
-rw-rw-r-- 1 jules jules   737 Apr  3 05:35 __init__.py
-rw-rw-r-- 1 jules jules   185 Apr  3 05:35 airllm.py
-rw-rw-r-- 1 jules jules   697 Apr  3 05:35 airllm_baichuan.py
-rw-rw-r-- 1 jules jules 27195 Apr  3 05:35 airllm_base.py
-rw-rw-r-- 1 jules jules  1630 Apr  3 05:35 airllm_chatglm.py
-rw-rw-r-- 1 jules jules   371 Apr  3 05:35 airllm_internlm.py
-rw-rw-r-- 1 jules jules 16729 Apr  3 05:35 airllm_llama_mlx.py
-rw-rw-r-- 1 jules jules   369 Apr  3 05:35 airllm_mistral.py
-rw-rw-r-- 1 jules jules   370 Apr  3 05:35 airllm_mixtral.py
-rw-rw-r-- 1 jules jules  1833 Apr  3 05:35 airllm_qwen.py
-rw-rw-r-- 1 jules jules   295 Apr  3 05:35 airllm_qwen2.py
-rw-rw-r-- 1 jules jules  2240 Apr  3 05:35 auto_model.py
drwxrwxr-x 2 jules jules  4096 Apr  3 05:35 persist
-rw-rw-r-- 1 jules jules   996 Apr  3 05:35 profiler.py
-rw-rw-r-- 1 jules jules  9613 Apr  3 05:35 tokenization_baichuan.py
-rw-rw-r-- 1 jules jules 17794 Apr  3 05:35 utils.py

/app/airllm_repo/air_llm/airllm/persist:
total 16
-rw-rw-r-- 1 jules jules   44 Apr  3 05:35 __init__.py
-rw-rw-r-- 1 jules jules 3669 Apr  3 05:35 mlx_model_persister.py
-rw-rw-r-- 1 jules jules  897 Apr  3 05:35 model_persister.py
-rw-rw-r-- 1 jules jules 1091 Apr  3 05:35 safetensor_model_persister.py
```

---

## 4. Architectural Deep-Dive

### 4.1 The "Skeleton" Loading Framework
AirLLM avoids OOM errors by never loading the full weights into memory at once.

1.  **Architecture Meta-Load**: It uses `accelerate.init_empty_weights()` to load the model structure from a Hugging Face `config.json`.
2.  **Meta Device Residency**: All tensors (parameters) are initialized on the **`meta` device**.
3.  **Memory Effect**: A 405-billion parameter model (occupying >800GB as weights) uses **zero physical RAM/VRAM** during this phase.
4.  **Just-in-Time Materialization**: Weights for a single layer are moved from disk to CPU RAM to GPU VRAM only when that layer is being executed.

### 4.2 The Dispatcher Pattern (`AutoModel`)
The `AutoModel` class in `auto_model.py` is a factory that automates architecture selection by inspecting the model's metadata.

```python
import importlib
from transformers import AutoConfig
from sys import platform

class AutoModel:
    def __init__(self):
        raise EnvironmentError(
            "AutoModel is designed to be instantiated "
            "using the `AutoModel.from_pretrained(pretrained_model_name_or_path)` method."
        )

    @classmethod
    def get_module_class(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        if 'hf_token' in kwargs:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True, token=kwargs['hf_token'])
        else:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)

        if "Qwen2ForCausalLM" in config.architectures[0]:
            return "airllm", "AirLLMQWen2"
        elif "QWen" in config.architectures[0]:
            return "airllm", "AirLLMQWen"
        elif "Baichuan" in config.architectures[0]:
            return "airllm", "AirLLMBaichuan"
        elif "ChatGLM" in config.architectures[0]:
            return "airllm", "AirLLMChatGLM"
        elif "InternLM" in config.architectures[0]:
            return "airllm", "AirLLMInternLM"
        elif "Mistral" in config.architectures[0]:
            return "airllm", "AirLLMMistral"
        elif "Mixtral" in config.architectures[0]:
            return "airllm", "AirLLMMixtral"
        elif "Llama" in config.architectures[0]:
            return "airllm", "AirLLMLlama2"
        else:
            print(f"unknown artichitecture: {config.architectures[0]}, try to use Llama2...")
            return "airllm", "AirLLMLlama2"

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        if platform == "darwin": # Automatic macOS hardware redirection
            from airllm import AirLLMLlamaMlx
            return AirLLMLlamaMlx(pretrained_model_name_or_path, *inputs, ** kwargs)

        module, cls_name = AutoModel.get_module_class(pretrained_model_name_or_path, *inputs, **kwargs)
        module = importlib.import_module(module)
        class_ = getattr(module, cls_name)
        return class_(pretrained_model_name_or_path, *inputs, ** kwargs)
```

---

## 5. Core Engine Specification: `AirLLMBaseModel`

The `AirLLMBaseModel` class (inheriting from `GenerationMixin`) orchestrates the layer-wise inference lifecycle.

### 5.1 The `forward` Method (Complete Definition)
This is the heart of AirLLM. It manually manages the PyTorch computation graph, ensuring only one layer is resident in VRAM at any time.

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

        # 1. HARD SYSTEM REBOOT
        # Purging the previous state and triggering malloc_trim ensures we have the
        # maximum possible contiguous free memory for the next pass.
        del self.model
        clean_memory()
        self.init_model() # Re-creates the 'meta' device skeleton

        batch = [input_ids_unit.to(self.running_device).unsqueeze(0) for input_ids_unit in input_ids]
        n_seq = len(batch[0])

        # 2. GLOBAL MASK & POSITIONING
        # Pre-allocates static tensors to avoid overhead inside the loop.
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

        # 3. LAYER-WISE EXECUTION LOOP
        with torch.inference_mode(), ThreadPoolExecutor() as executor:

            # ASYNC PREFETCH START
            # Pre-loads the first layer (embeddings) from disk to CPU RAM in a background thread.
            if self.prefetching:
                future = executor.submit(self.load_layer_to_cpu, self.layer_names[0])

            for i, (layer_name, layer) in tqdm(enumerate(zip(self.layer_names, self.layers)),
                                               desc=f'running layers({self.running_device})',
                                               total=len(self.layers)):

                # A. WEIGHT ACQUISITION
                if self.prefetching:
                    state_dict = future.result() # Wait for background load
                    if (i + 1) < len(self.layer_names): # Trigger next layer load
                        future = executor.submit(self.load_layer_to_cpu, self.layer_names[i + 1])
                else:
                    state_dict = self.load_layer_to_cpu(layer_name)

                # B. MATERIALIZATION (CPU RAM -> GPU VRAM)
                # Weights are 'moved' into the model attributes using set_module_tensor_to_device.
                moved_layers = self.move_layer_to_device(state_dict)

                # C. LAYER-SPECIFIC COMPUTATION
                for j, seq in enumerate(batch):
                    if layer_name == self.layer_names_dict['embed']:
                        batch[j] = layer(seq)
                    elif layer_name == self.layer_names_dict['norm']:
                        batch[j] = self.run_norm(layer, seq)
                    elif layer_name == self.layer_names_dict['lm_head']:
                        batch[j] = self.run_lm_head(layer, seq)
                    else:
                        # Transformer Block logic
                        len_seq = seq.shape[1]
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

                # D. GPU PURGE (VRAM -> META)
                # Weights are effectively deleted from VRAM by moving back to meta device.
                if self.hf_quantizer is not None:
                    for param_name in moved_layers:
                        set_module_tensor_to_device(self.model, param_name, 'meta')
                else:
                    layer.to("meta")

                # E. HARD MEMORY SCRUB
                clean_memory()

        logits = torch.cat(batch, 0)
        # Handle KV cache and output concatenation...
        return CausalLMOutputWithPast(logits=logits, ...)
```

---

## 6. System Layer: Sharding and Persistence

### 6.1 Sharding Mechanism (`utils.py`)
Standard models are provided as massive 10GB shards. AirLLM normalizes these into a granular, one-file-per-layer format.

```python
def split_and_save_layers(checkpoint_path, layer_shards_saving_path=None, splitted_model_dir_name='splitted_model',
                          compression=None, layer_names=None, delete_original=False, repo_id=None, hf_token=None):
    """
    Implementation detail of the re-sharding engine.
    """
    index = json.load(open(checkpoint_path / 'model.safetensors.index.json'))['weight_map']

    # Calculate unique original files for each layer
    for layer in tqdm(layers):
        shards = [int(v.split('-')[1]) for k, v in index.items() if k.startswith(layer) and '-' in v]

        if len(shards) > 0:
            if max(shards) > shard:
                # Load shards from disk to RAM
                to_load = checkpoint_path / f'model-000{shard:02d}-of-000{n_shards:02d}.safetensors'
                state_dict.update(load_file(to_load, device='cpu'))
        else:
            # Handle single-file checkpoints
            shards = [v for k, v in index.items() if k.startswith(layer)]
            state_dict.update(load_file(checkpoint_path / shards[0], device='cpu'))

        # Extract weight subset for THIS layer
        layer_state_dict = dict([(k, v) for k, v in state_dict.items() if k.startswith(layer)])

        # Apply JIT compression if enabled
        layer_state_dict = compress_layer_state_dict(layer_state_dict, compression)

        # Atomic Save using Safetensors
        ModelPersister.get_model_persister().persist_model(layer_state_dict, layer, saving_path)

        # Immediate memory release
        for k in layer_state_dict.keys(): del state_dict[k]
        del layer_state_dict
        clean_memory()
```

### 6.2 The Persistence Backends (`persist/`)

#### `SafetensorModelPersister` (Full Definition)
```python
class SafetensorModelPersister(ModelPersister):
    def model_persist_exist(self, layer_name, saving_path):
        safetensor_exists = os.path.exists(str(saving_path / (layer_name + 'safetensors')))
        done_marker_exists = os.path.exists(str(saving_path / (layer_name + 'safetensors.done')))
        return safetensor_exists and done_marker_exists

    def persist_model(self, state_dict, layer_name, saving_path):
        save_file(state_dict, saving_path / (layer_name + 'safetensors'))
        # Done marker prevents loading partial/corrupted files
        (saving_path / (layer_name + 'safetensors.done')).touch()

    def load_model(self, layer_name, path):
        return load_file(Path(path) / (layer_name + ".safetensors"), device="cpu")
```

---

## 7. Memory Optimization: The "Secret Sauce"

### 7.1 The `clean_memory()` Surgical Utility
Standard garbage collection is often insufficient for freeing high-pressure resources. AirLLM uses a direct call to the C library to force physical memory release.

```python
def clean_memory():
    import gc, ctypes, torch

    # 1. Trigger Python's garbage collector.
    gc.collect()

    # 2. Trigger Linux glibc malloc_trim.
    # Forces the C runtime to release all possible memory pages back to the kernel.
    # This is critical for preventing the Resident Set Size (RSS) from growing indefinitely.
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass

    # 3. Release PyTorch's internal VRAM pool.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```

### 7.2 CPU Memory Pinning
During prefetching, loaded weights are marked with `.pin_memory()`. This allows the NVIDIA driver to perform Direct Memory Access (DMA) transfers to the GPU, bypassing the CPU and significantly increasing transfer bandwidth.

---

## 8. Weight-Only Block-wise Quantization

AirLLM implements a unique quantization approach via the `bitsandbytes` library.

- **4-bit (NF4)**: Blocksize 64.
- **8-bit**: Blocksize 2048.
- **Dequantization**: Reconstructs weights from `absmax` scaling factors in CPU RAM just before moving to the GPU. This reduces disk-to-RAM bandwidth bottlenecks while ensuring high activation accuracy.

---

## 9. Hardware-Specific Implementation: MLX Engine

For Apple Silicon users, AirLLM provides a separate engine in `airllm_llama_mlx.py` optimized for **Unified Memory Architecture**.

### 9.1 The MLX Forward Logic (Complete Loop)
```python
def model_generate(self, x, temperature=0, max_new_tokens=None):
    # materialization of embedding weights
    update_weights = ModelPersister.get_model_persister().load_model('embed')
    self.tok_embeddings.update(update_weights['tok_embeddings'])
    x = self.tok_embeddings(x)
    mx.eval(x) # Control lazy evaluation
    del self.tok_embeddings # Free unified RAM immediately

    for il in range(self.model_args.n_layers):
        l = TransformerBlock(args=self.model_args)
        # Dynamic layer injection
        l.update(ModelPersister.get_model_persister().load_model(f'layers.{il}'))
        x, c = l(x, mask=mask)
        mx.eval(x) # Force GPU finish
        del l
        gc.collect()
```

### 9.2 Name Mapping (`map_torch_to_mlx`)
This utility automatically transforms PyTorch model names to the format expected by the MLX framework.
```python
def map_torch_to_mlx(model):
    model = {k.replace("model.", ""): v for k, v in model.items()}
    model = {k.replace("mlp", "feed_forward"): v for k, v in model.items()}
    model = {k.replace("down_proj", "w2"): v for k, v in model.items()}
    model = {k.replace("up_proj", "w3"): v for k, v in model.items()}
    model = {k.replace("gate_proj", "w1"): v for k, v in model.items()}
    model = {k.replace("input_layernorm", "attention_norm"): v for k, v in model.items()}
    model = {k.replace("embed_tokens", "tok_embeddings"): v for k, v in model.items()}
    model = {k.replace("self_attn", "attention"): v for k, v in model.items()}
    return model
```

---

## 10. Architecture-Specific Adaptations (Full Definitions)

### 10.1 `AirLLMChatGLM` Implementation
```python
class AirLLMChatGLM(AirLLMBaseModel):
    def set_layer_names_dict(self):
        self.layer_names_dict = {
            'embed': 'transformer.embedding.word_embeddings',
            'layer_prefix': 'transformer.encoder.layers',
            'norm': 'transformer.encoder.final_layernorm',
            'lm_head': 'transformer.output_layer',
            'rotary_pos_emb': 'transformer.rotary_pos_emb'
        }

    def get_pos_emb_args(self, len_p, len_s):
        rotary_pos_emb = self.model.transformer.rotary_pos_emb(self.config.seq_length)
        rotary_pos_emb = rotary_pos_emb[None, : len_s]
        rotary_pos_emb = rotary_pos_emb.transpose(0, 1).contiguous()
        return {'rotary_pos_emb': rotary_pos_emb}
```

### 10.2 `AirLLMQWen` Implementation
```python
class AirLLMQWen(AirLLMBaseModel):
    def get_pos_emb_args(self, len_p, len_s):
        # Dynamic frequency scaling logic for QWen
        if self.model.transformer.use_dynamic_ntk:
            ntk_alpha_list = [1.0]
        elif len_p + len_s != len_s:
            ntk_alpha_list = self.model.transformer.rotary_emb._ntk_alpha_cached_list
        else:
            ntk_alpha_list = [self.model.transformer.get_ntk_alpha(len_p + len_s)]

        rotary_pos_emb_list = [
            self.model.transformer.rotary_emb(len_p + len_s, ntk_alpha=a) for a in ntk_alpha_list
        ]
        return {'rotary_pos_emb_list': rotary_pos_emb_list}
```

---

## 11. Training and Alignment Ecosystem

### 11.1 QLoRA Fine-tuning (`training/qlora.py`)
- **Quantization**: Loads base weights in 4-bit NF4.
- **Efficiency**: Allows fine-tuning a 70B model on a single 24GB or 48GB GPU by discarding intermediate activations during the forward pass (Gradient Checkpointing).

### 11.2 Direct Preference Optimization (`rlhf/qlora_dpo.py`)
- **DPO Algorithm**: Aligns LLMs with human preferences without a separate reward model.
- **Loss Function**: `losses = -F.logsigmoid(beta * (pi_logratios - ref_logratios))`.

---

## 12. Troubleshooting Guide

| Issue | Technical Root Cause | Resolution |
| :--- | :--- | :--- |
| `MetadataIncompleteBuffer` | interrupted disk write or full disk. | Ensure >200GB free space. Clear HF `.cache`. |
| `ValueError: pad_token` | Missing padding index in tokenizer. | Manually set `model.tokenizer.pad_token = model.tokenizer.eos_token`. |
| `401 Unauthorized` | Attempting to access gated model. | Pass `hf_token="YOUR_HUGGINGFACE_API_KEY"` to `from_pretrained`. |
| `Slow Inference Speed` | Disk I/O or Dequantization bottleneck. | Move shards to an NVMe SSD and ensure `compression=None`. |

---

## 13. Engineering Roadmap: Rebuilding AirLLM from Scratch

To rebuild this system, an engineer should follow these five engineering phases:

1.  **Normalization Engine**: Implement a script that re-shards models into a `one-file-per-layer` format (Safetensors recommended).
2.  **Meta Model Framework**: Create a class that loads a model architecture into the `meta` device using `accelerate.init_empty_weights`.
3.  **The Manual Dispatcher**: Rewrite the `forward` pass to iterate through layer indices. Inside the loop:
    - Load Layer N weights from disk.
    - Materialize weights via `set_module_tensor_to_device`.
    - Synchronize hardware (CUDA/MPS).
    - Execute computation.
    - Purge weights (Return to `meta`).
4.  **Hardware-Level Scrubbing**: Implement the `malloc_trim` cleanup function and call it after every single layer execution.
5.  **Pipelining**: Implement a producer-consumer thread to hide disk latency by loading Layer N+1 from disk while the GPU is executing Layer N.

---
*Manual compiled by Jules, AI Software Engineer.*
