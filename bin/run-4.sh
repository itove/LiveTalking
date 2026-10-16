#!/bin/bash
#
# vim:ft=bash

############### Variables ###############

############### Functions ###############

############### Main Part ###############

export WAKE_ENABLE=1
export WAKE_WORD='杰克|捷克|jack|Jack'
export WAKE_WINDOW=25

python app.py --transport webrtc --model wav2lip --avatar_id wangshuji \
    --listenport 8020 \
    --llm_provider openai \
    --llm_base_url http://172.20.29.123:4000/v1 \
    --llm_model qwen3.8-27b \
    --tts omnitts \
    --ASR_SERVER http://172.20.116.82:8010 --asr_model qwen3-asr-1.7b \
    --TTS_SERVER http://172.20.116.82:8011 --REF_FILE 'wang_xu_pei'
