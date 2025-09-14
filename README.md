https://colab.research.google.com/drive/1iTZfNal6bX7UiFny5M1Ttpgi7JiPToFz?usp=sharing

# classification
      python train.py \
      --model resnet \
      --data_dir ./data \
      --dataset cifar10 \
      --batch_size 128 \
      --epochs 50 \
      --learning_rate 0.1 \
      --weight_decay 5e-4 \
      --momentum 0.9 \
      --seed 42 \
      --use_aug \
      --scheduler cosine \
      --optimizer SGD \
      --loss CE \
      --use_early_stopping \
      --patience 20 \
      --img_size 224 \
      --num_workers 4 \
      --use_amp \
      --ckpt_dir ./checkpoints \
      --num_classes 10
