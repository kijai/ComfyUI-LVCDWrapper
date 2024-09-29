# ComfyUI wrapper nodes for LVCD


https://github.com/user-attachments/assets/6d4d5afd-f8b1-499b-8872-d5d2f6929da4

Requires SVD model, seems to work best with the original one, but runs with 1.1 and XT as well, this is loaded normally from ComfyUI/models/checkpoints:

https://huggingface.co/stabilityai/stable-video-diffusion-img2vid

fp16 version:

https://huggingface.co/Kijai/LVCD-pruned/blob/main/svd-fp16.safetensors

LVCD model itself goes to ComfyUI/models/lvcd (autodownloaded if it doesn't exist):

https://huggingface.co/Kijai/LVCD-pruned/blob/main/lvcd-fp16.safetensors

Original repo:

https://github.com/luckyhzt/LVCD
