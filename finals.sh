alpha=0.9

temp=1.0

loss_meth="method1"

learing_rates="0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09"

for lr in $learing_rates
do
    echo '[KD] alpha: ' $alpha ', temperature: ' $temp
    python main.py --KD --model student --lr $lr --alpha $alpha --temperature $temp --epochs 60 --loss_method $loss_meth 
done
