for alpha in 0.1
do 
for model in random_forest
do

python run_full_training.py \
    --alpha $alpha \
    --model_name $model \
    -th 15 \
    -w 24 \
    --max_depth 6 \
    --n_estimators 800 \
    --early_stopping_rounds 200 \
    --run_name $model\_alpha_$alpha\_fullfeatures \
    --eval_metric auc \
    --scaler robust \
    --false_alert_budget_per_month 2.0 \
    --false_alert_cost 0.05 \
    --min_lead_hours 1 \
    --min_lead_utility 0.10 \
    --utility_power 1.0 \
    --max_lead_hours 24 \
    --utility_target_lead_hours 24 \
    --alert_cooldown_hours 24

done
done
