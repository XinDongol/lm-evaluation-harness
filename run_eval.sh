# accelerate launch -m 
lm_eval --model xinmodel \
    --model_args is_hf=False,tokenizer=/lustre/fsw/portfolios/nvr/users/xind/miscs/models/Qwen2.5-0.5B,model_name=qwen_block_distill,model_flavor=qwen1p5b_avgpool2_repeat2shift1_7-14-7,checkpoint_path=/lustre/fsw/portfolios/nvr/users/xind/megatron_exp/qwen1p5b_avgpool2_repeat2shift1_7-14-7_lr1e4/iter_0090000/mp_rank_00/model_optim_rng.pt \
    --tasks piqa \
    --device cuda:0 \
    --batch_size 1 \
    --seed 42 \
    --wandb_args project=lm-eval-harness,name=qwen1p5b_avgpool2_repeat2shift1_7-14-7_lr1e4_1gpu \
    --log_samples \
    --output_path /lustre/fs12/portfolios/nvr/users/xind/megatron_exp/eval_results/qwen1p5b_avgpool2_repeat2shift1_7-14-7_lr1e4
