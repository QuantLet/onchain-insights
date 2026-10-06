for alpha in 0.4 1.0
do
for task in distribution
do

python main_lightning.py \
    --alpha $alpha \
    --forecast_task $task \
    --model_name iTransformer \
    --method forecast \
    --experiment_name stablecoin-depeg \
    --run_name "alpha_${alpha}_${task}" \
    --n_epochs 50 \
    --patience 10 \
    --verbose 1 \
    --check_lr \
    --seq_len  168 \
    --pred_len 24 \
    --val_split 0.7 \
    --test_split 0.85 \
    --batch_size 256 \
    --test_batch_size 20 \
    --scaler revin \
    --affine 1 \
    --remote_logging \
    --n_cheb 8 \
    --twcrps_side two_sided \
    --twcrps_smooth_h 2 \
    --u_grid_size 256 \

done
done