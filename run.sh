
CRYPTOGRAPHY_OPENSSL_NO_LEGACY=1 python -m tensorboard.main --logdir=logs/ZH_MODEL/

nohup python -m torch.distributed.launch --master_port 62580 --nproc-per-node=1 --use_env train_ms.py -c /root/autodl-tmp/models/Bert-VITS2-HBAI/configs/config.zh.enzh.json -m ALL_ZH_MODEL_V2 >> logs/all.zh.enzh.log 2>&1 &

nohup python -m torch.distributed.launch --master_port 62570 --nproc-per-node=1 --use_env train_ms.py -c /root/autodl-tmp/models/Bert-VITS2-HBAI/configs/config.zh.enzh.large.json -m ALL_ZH_MODEL_V4 >> logs/all.zh.log.v4 2>&1 &
