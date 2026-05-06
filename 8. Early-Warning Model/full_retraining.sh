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

done
done