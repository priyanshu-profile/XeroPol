#!/bin/bash

model_type=$2


if [ $1 == "emowoz_english" ]
then
    python main.py --task emowoz \
                   --train_languages en \
                   --dev_languages en \
                   --test_languages en \
                   --model_dir emowoz_english \
                   --do_train \
                   --do_eval \
                   --cuda_device cuda:0 \
                   --train_batch_size 16 \
                   --eval_batch_size 32 \
                   --gradient_accumulation_steps 5 \
                   --num_train_epochs 30 \
                   --learning_rate 0.00002 \
                   --save_model \
                   --model_type $2 \
                   --max_seq_len 180
fi

if [ $1 == "emowoz_aligned" ]
then
    python main.py --task emowoz \
                       --train_languages en \
                       --dev_languages hi \
                       --test_languages hi \
                       --model_dir emowoz_aligned_hi \
                       --do_train \
                       --do_eval \
                       --cuda_device cuda:0 \
                       --train_batch_size 16 \
                       --eval_batch_size 32 \
                       --gradient_accumulation_steps 5 \
                       --num_train_epochs 30 \
                       --learning_rate 0.00002 \
                       --align_languages hi \
                       --save_model \
                       --model_type $2 \
                       --max_seq_len 180
					   --aux_loss scl
					   --align_loss cal
    
fi

if [ $1 == "emowoz_zero_shot" ]
then
    python main.py --task emowoz \
                       --train_languages hi \
                       --dev_languages hi \
                       --test_languages hi \
                       --model_dir emowoz_zero_shot_hi \
                       --do_eval \
                       --cuda_device cuda:0 \
                       --eval_batch_size 32 \
                       --model_type $2 \
                       --load_eval_model emowoz_english \
                       --max_seq_len 180
					   
fi

if [ $1 == "emowoz_target" ]
then
    python main.py --task emowoz \
                       --train_languages hi \
                       --dev_languages hi \
                       --test_languages hi \
                       --model_dir emowoz_target_hi \
                       --do_eval \
                       --do_train \
                       --cuda_device cuda:0 \
                       --train_batch_size 16 \
                       --eval_batch_size 32 \
                       --gradient_accumulation_steps 5 \
                       --num_train_epochs 30 \
                       --learning_rate 0.00002 \
                       --model_type $2 \
                       --max_seq_len 180
					   --aux_loss scl

fi

if [ $1 == "emowoz_eval" ]
then
    python main.py --task emowoz \
                       --train_languages hi \
                       --dev_languages hi \
                       --test_languages hi \
                       --model_dir emowoz_eval_hi \
                       --do_eval \
                       --cuda_device cuda:0 \
                       --eval_batch_size 32 \
                       --load_eval_model emowoz_aligned_all \
                       --model_type $2 \
                       --max_seq_len 180
    done
fi
