#!/bin/bash
#
# vim:ft=bash

############### Variables ###############

############### Functions ###############

############### Main Part ###############

python app.py --transport webrtc --model wav2lip --avatar_id wav2lip256_avatar1 \
    --llm_provider openai \
    --llm_base_url http://172.20.84.107:8002/v1 \
    --llm_model qwen3.8-27b
