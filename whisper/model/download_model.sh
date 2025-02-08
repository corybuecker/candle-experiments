#!/bin/bash

set -e
set -u
set -o pipefail

IFS=$'\n\t'


rm model.safetensors config.json tokenizer.json
huggingface-cli download --local-dir . openai/whisper-small.en model.safetensors config.json tokenizer.json
