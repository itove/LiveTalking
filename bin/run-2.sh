#!/bin/bash
#
# vim:ft=bash

############### Variables ###############

############### Functions ###############

############### Main Part ###############

python app.py --transport webrtc --model wav2lip --avatar_id wav2lip256_avatar1 \
    --listenport 8020 \
    --llm_provider openai \
    --llm_base_url http://172.20.29.123:4000/v1 \
    --llm_model qwen3.8-27b \
    --tts omnitts \
    --TTS_SERVER http://172.20.116.82:8011 --REF_FILE 'wang_xu_pei'
