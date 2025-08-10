# vLLM Setup Guide - Network Connectivity Solutions

## Your Current Setup Issue
You're encountering: `OSError: We couldn't connect to 'https://huggingface.co'`

This happens when vLLM can't download the model from Hugging Face.

## Quick Fixes

### 1. Test Your Token & Internet
```bash
# Check if your token works
curl -H "Authorization: Bearer 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" \
  https://huggingface.co/api/whoami

# Check model access
curl -H "Authorization: Bearer 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" \
  https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
```

### 2. Use Public Model First (No Token Needed)
```bash
docker run -it --rm -p 8000:8000 \
  vllm/vllm-openai:latest \
  --model microsoft/DialoGPT-medium \
  --dtype float16
```

### 3. Pre-download Model (Recommended)
```bash
# Step 1: Download model to cache
docker run -it --rm \
  --env HUGGING_FACE_HUB_TOKEN=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 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  huggingface/transformers-pytorch-gpu \
  python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B-Instruct'); AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.2-1B-Instruct')"

# Step 2: Run vLLM with cached model
docker run -it --rm -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:latest \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --dtype float16
```

### 4. Offline Mode
```bash
docker run -it --rm -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e TRANSFORMERS_OFFLINE=1 \
  -e HF_DATASETS_OFFLINE=1 \
  vllm/vllm-openai:latest \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --dtype float16
```

### 5. Run Troubleshooting Script
```bash
python start_with_vllm.py
```

## Testing Your Setup
```bash
# Check server health
curl http://localhost:8000/health

# List models
curl http://localhost:8000/v1/models

# Test completion
curl -X POST "http://localhost:8000/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.2-1B-Instruct",
    "prompt": "Hello, how are you?",
    "max_tokens": 50
  }'
```

## Alternative Models (No Token Required)
- `microsoft/DialoGPT-medium` - Conversational AI
- `gpt2` - Classic GPT-2
- `facebook/opt-125m` - Small OPT model

Reference: https://www.datacamp.com/tutorial/vllm