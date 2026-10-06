
for alpha in 0.1
do
for task in distribution
do
for model in ANGEL
do
for loss in combined
do
for tail in gpd
do
for decomp in spline
do
for revin in revin
do
for knot_p in 3.0
do
for grid in power-tail
do
for selector in sparsemax
do
for gate in 1
do
for lam in 0
do
for h in 1
do
python main_lightning.py \
    --alpha $alpha \
    --forecast_task $task \
    --model_name $model \
    --method forecast \
    --experiment_name stablecoin-paper \
    --run_name "ANGEL_full_${selector}" \
    --n_epochs 100 \
    --patience 5 \
    --verbose 1 \
    --seq_len 168 \
    --pred_len 24 \
    --val_split 0.55 \
    --test_split 0.7 \
    --batch_size 15 \
    --test_batch_size 20 \
    --revin_type $revin \
    --affine 1 \
    --dist_loss $loss \
    --n_cheb 6 \
    --twcrps_side two_sided \
    --twcrps_smooth_h $h \
    --u_grid_size 256 \
    --grid_density $grid \
    --quantile_decomp $decomp \
    --knot_kind $grid \
    --knot_p $knot_p \
    --spline_degree 3 \
    --tail_model $tail \
    --gpd_u_low 0.1 \
    --gpd_u_high 0.9 \
    --gpd_xi_min 0.0 \
    --gpd_xi_max 1 \
    --remote_logging \
    --selector_activation $selector \
    --attn_activation softmax \
    --l0_lambda $lam \
    --fusion_n_layers 2 \
    --fusion_n_heads 2 \
    --cov_n_layers 2 \
    --cov_n_heads 4 \
    --learning_rate 1e-5 \
    --rho_combined 0.7 \
    --rho_schedule none \
    --save_test_diagnostics 1 \
    --d_model 256 \
    --cov_d_ff 256 \
    --selector_hidden 128 \
    --fusion_d_ff 256 \
    --dropout 0.0796 \

done
done
done
done
done
done
done
done
done
done
done
done
done
