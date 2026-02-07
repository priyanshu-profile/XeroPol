#!/bin/bash

model_type=$2


if [ $1 == "multidogo_english" ]
then
    python main.py --task multidogo \
                   --train_languages en \
                   --dev_languages en \
                   --test_languages en \
                   --model_dir multidogo_english \
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
                   --max_seq_len 160
fi

if [ $1 == "multidogo_aligned" ]
then
    python main.py --task multidogo \
                       --train_languages en \
                       --dev_languages hi \
                       --test_languages hi \
                       --model_dir multidogo_aligned_hi \
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
                       --max_seq_len 160
					   --aux_loss scl
					   --align_loss cal
    
fi

if [ $1 == "multidogo_zero_shot" ]
then
    python main.py --task multidogo \
                       --train_languages hi \
                       --dev_languages hi \
                       --test_languages hi \
                       --model_dir multidogo_zero_shot_hi \
                       --do_eval \
                       --cuda_device cuda:0 \
                       --eval_batch_size 32 \
                       --model_type $2 \
                       --load_eval_model multidogo_english \
                       --max_seq_len 160
					   
fi

if [ $1 == "multidogo_target" ]
then
    python main.py --task multidogo \
                       --train_languages hi \
                       --dev_languages hi \
                       --test_languages hi \
                       --model_dir multidogo_target_hi \
                       --do_eval \
                       --do_train \
                       --cuda_device cuda:0 \
                       --train_batch_size 16 \
                       --eval_batch_size 32 \
                       --gradient_accumulation_steps 5 \
                       --num_train_epochs 30 \
                       --learning_rate 0.00002 \
                       --model_type $2 \
                       --max_seq_len 160
					   --aux_loss scl

fi

if [ $1 == "multidogo_eval" ]
then
    python main.py --task multidogo \
                       --train_languages hi \
                       --dev_languages hi \
                       --test_languages hi \
                       --model_dir multidogo_eval_hi \
                       --do_eval \
                       --cuda_device cuda:0 \
                       --eval_batch_size 32 \
                       --load_eval_model multidogo_aligned_all \
                       --model_type $2 \
                       --max_seq_len 160
    done
fi
