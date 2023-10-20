ZH: /usr/bin/env /root/autodl-tmp/miniconda3/envs/bert-vits2/bin/python /root/autodl-tmp/miniconda3/envs/bert-vits2/lib/python3.10/site-packages/torch/distributed/launch.py --nproc-per-node=1 --use_env train_ms.py -c /root/autodl-tmp/models/Bert-VITS2-HBAI/configs/config.zh.json
JP: /usr/bin/env /root/autodl-tmp/miniconda3/envs/bert-vits2/bin/python /root/autodl-tmp/miniconda3/envs/bert-vits2/lib/python3.10/site-packages/torch/distributed/launch.py --nproc-per-node=1 --use_env --master_port 62580 train_ms.py -c /root/autodl-tmp/models/Bert-VITS2-HBAI/configs/config.jp.json -m JP_MODEL 

CE: python -m torch.distributed.launch --nproc-per-node=1 --use_env train_ms.py -c /root/autodl-tmp/models/Bert-VITS2-HBAI/configs/config.ce.json -m CE_MODEL

CRYPTOGRAPHY_OPENSSL_NO_LEGACY=1 python -m tensorboard.main --logdir=logs/ZH_MODEL/
