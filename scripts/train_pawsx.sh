#!/bin/bash
# Copyright 2020 Google and DeepMind.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

REPO=$PWD
MODEL=${1:-bert-base-multilingual-cased}
GPU=${2:-0}
Seed=${3:-1}
DATA_DIR=${4:-"$REPO/download/"}
OUT_DIR=${5:-"$REPO/outputs/"}

export CUDA_VISIBLE_DEVICES=$GPU

TASK='pawsx'
LR=3e-5
EPOCH=3
MAXL=128
LANGS="de,en,es,fr,ja,ko,zh"
LC=""
if [ $MODEL == "bert-base-multilingual-cased" ]; then
  MODEL_TYPE="bert"
elif [ $MODEL == "xlm-mlm-100-1280" ] || [ $MODEL == "xlm-mlm-tlm-xnli15-1024" ]; then
  MODEL_TYPE="xlm"
  LC=" --do_lower_case"
elif [ $MODEL == "xlm-roberta-large" ] || [ $MODEL == "xlm-roberta-base" ]; then
  MODEL_TYPE="xlmr"
fi

if [ $MODEL == "xlm-roberta-large" ]; then
  LR=5e-6
fi

if [ $MODEL == "xlm-mlm-100-1280" ] || [ $MODEL == "xlm-roberta-large" ]; then
  BATCH_SIZE=16
  GRAD_ACC=2
  
else
  LR=2e-5
  BATCH_SIZE=32
  GRAD_ACC=1
fi


SAVE_DIR="${OUT_DIR}/${TASK}/${MODEL}-LR${LR}-epoch${EPOCH}-MaxLen${MAXL}-Seed${Seed}-0.5*codecos-0.5ratio-5e-6-5e-6-codelog-0.5w/"
mkdir -p $SAVE_DIR

nohup python $PWD/third_party/run_classify_2stage.py \
  --model_type $MODEL_TYPE \
  --model_name_or_path $MODEL \
  --train_language en \
  --task_name $TASK \
  --do_train \
  --do_eval \
  --do_predict \
  --seed $Seed \
  --train_split train \
  --test_split test \
  --data_dir $DATA_DIR/$TASK/ \
  --gradient_accumulation_steps $GRAD_ACC \
  --save_steps 200 \
  --per_gpu_train_batch_size $BATCH_SIZE \
  --learning_rate $LR \
  --num_stage1_epochs 2 \
  --num_train_epochs $EPOCH \
  --max_seq_length $MAXL \
  --output_dir $SAVE_DIR \
  --eval_all_checkpoints \
  --overwrite_output_dir \
  --overwrite_cache \
  --log_file 'train.log' \
  --predict_languages $LANGS \
  --save_only_best_checkpoint $LC \
  --eval_test_set >pawsx.$MODEL_TYPE.out.log &

