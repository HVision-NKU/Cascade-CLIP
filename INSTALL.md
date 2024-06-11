## Installation

```bash
conda create --name cascade_clip python=3.7 -y
conda activate cascade_clip
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
pip install openmim
mim install mmcv-full==1.5.0
pip install mmsegmentation==0.24.0
pip install -r requirements.txt
```

>If the following error occurs, you can directly replace from **torch._six import container_abcs** with import **collections.abc as container_abcs** in
>envs/cascade_clip/lib/python3.7/site-packages/timm/models/layers/helpers.py

>ImportError: cannot import name 'container_abcs' from 'torch._six' (/envs/cascade_clip/lib/python3.7/site-packages/>torch/_six.py)
>    from torch._six import container_abcs




- Also directly apply the image provided by [ZegCLIP](https://github.com/ZiqinZhou66/ZegCLIP) in Dockerhub:


 `docker push ziqinzhou/zegclip:latest`