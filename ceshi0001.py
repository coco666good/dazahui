import argparse
import binascii

# 自定义类型转换函数
def str_to_bool(value):
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Invalid boolean value: {}'.format(value))

# 创建参数解析器
parser = argparse.ArgumentParser()

parser.add_argument('--UI', type=str)
parser.add_argument('--Version', type=str)
parser.add_argument('--ControlNet', type=str)
parser.add_argument('--Drive_Map', type=str_to_bool)
parser.add_argument('--Key_words', type=str_to_bool)


# 解析命令行参数
args = parser.parse_args()


UI = args.UI
Version = args.Version
ControlNet = args.ControlNet
Drive_Map = args.Drive_Map
Key_words = args.Key_words


################################################################################################################################################

import sys
import os
import base64
import importlib.util
from IPython import get_ipython
from IPython.display import clear_output
from google.colab import drive
import tensorflow as tf

# 检测是否为GPU运行
print("TensorFlow version:", tf.__version__)
if tf.test.gpu_device_name():
    drive.mount('/content/drive')
else:
    raise Exception("\n请在《代码执行程序》-《更改运行时类型》-设置为GPU~")

w = base64.b64decode(("d2VidWk=").encode('ascii')).decode('ascii')
sdw = base64.b64decode(("c3RhYmxlLWRpZmZ1c2lvbi13ZWJ1aQ==").encode('ascii')).decode('ascii')
# sdw = binascii.unhexlify("737461626c652d646966667573696f6e2d7765627569").decode('ascii')
# w = binascii.unhexlify("7765627569").decode('ascii')
wb = f'/content/{sdw}'
gwb = f'/content/drive/MyDrive/GUA'

get_ipython().run_line_magic('cd', '/content')
get_ipython().run_line_magic('env', 'TF_CPP_MIN_LOG_LEVEL=1')

# 云盘同步
def cloudDriveSync(cloudPath, localPath='', sync=False):
    # 云盘没有目录
    if not os.path.exists(cloudPath):
        # 创建云盘目录
        get_ipython().system(f'mkdir {cloudPath}')
    
    # 是否要同步
    if not sync:
        return
    
    # 删除本地目录
    get_ipython().system(f'rm -rf {localPath}')
    # 链接云盘目录
    get_ipython().system(f'ln -s {cloudPath} {localPath}')
    
# 初始化云盘
def initCloudDrive():
    cloudDriveSync(f'{gwb}')
    cloudDriveSync(f'{gwb}/Config')
    cloudDriveSync(f'{gwb}/Models', f'{wb}/models/Stable-diffusion', Drive_Map)
    cloudDriveSync(f'{gwb}/Lora', f'{wb}/models/Lora', Drive_Map)
    cloudDriveSync(f'{gwb}/LyCORIS', f'{wb}/models/LyCORIS', Drive_Map)
    cloudDriveSync(f'{gwb}/hypernetworks', f'{wb}/models/hypernetworks', Drive_Map)
    cloudDriveSync(f'{gwb}/Vae', f'{wb}/models/VAE', Drive_Map)
    cloudDriveSync(f'{gwb}/Outputs', f'{wb}/outputs', Drive_Map)

    # 云盘没有配置文件
    if not os.path.exists(f'{gwb}/Config/config.json'):
        get_ipython().system(f'wget -O {gwb}/Config/config.json "https://huggingface.co/wageguagua/main/raw/main/config.json"')

# clong git
def gitDownload(url, localPath):
    if os.path.exists(localPath):
        return
    
    get_ipython().system(f'git clone {url} {localPath}')

# 安装附加功能
def installAdditional():
    # 安装扩展
    urls = [
        f'https://github.com/camenduru/{sdw}-images-browser',              # 图像浏览器
        f'https://github.com/camenduru/sd-{w}-tunnels',                    # Tunnel 网络支持
        f'https://github.com/etherealxx/batchlinks-{w}',                   # 批量下载模型lora
        f'https://github.com/camenduru/sd-civitai-browser',                # Civitai 分类和搜索
        f'https://github.com/KohakuBlueleaf/a1111-sd-{w}-lycoris',         # 人物生成模型
        f'https://github.com/AUTOMATIC1111/{sdw}-rembg',                   # 背景移除功能
        f'https://github.com/thomasasfk/sd-{w}-aspect-ratio-helper',       # 宽高比调整功能
        f'https://github.com/kohya-ss/sd-{w}-additional-networks',         # 模型网络
        f'https://github.com/fkunn1326/openpose-editor',                   # 人体姿态编辑功能
        f'https://github.com/jexom/sd-{w}-depth-lib',                      # 深度图编辑器
        f'https://github.com/hnmr293/posex',                               # 人体姿态估计
        f'https://github.com/s9roll7/ebsynth_utility',                     # 视频的图像生成
        f'https://github.com/ashen-sensored/{sdw}-two-shot',               # 潜变量成对(双人特写)
        f'https://github.com/nonnonstop/sd-{w}-3d-open-pose-editor',       # 3D 人体姿态编辑功能
        f'https://github.com/camenduru/{sdw}-huggingface',                 # 整合 Huggingface 的模型功能
        f'https://github.com/camenduru/{sdw}-catppuccin',                  # Catppuccin 主题
        f'https://github.com/hanamizuki-ai/{sdw}-localization-zh_Hans',    # 中文
        f'https://github.com/numz/sd-wav2lip-uhq',                         # 数字人说话
        f'https://github.com/IDEA-Research/DWPose',                        # cnt手部模型
        # f'https://github.com/a2569875/{sdw}-composable-lora',              # 可自组 LoRA
        # f'https://github.com/deforum-art/deforum-for-automatic1111-{w}',   # Deforum动态视频创作
        # f'https://github.com/Scholar01/sd-{w}-mov2mov',                    # 视频转换
        # f'https://github.com/tjm35/asymmetric-tiling-sd-{w}',              # 非对称平铺功能
        # f'https://github.com/hako-mikan/sd-{w}-lora-block-weight',         # LORA 模型的权重调整
        # f'https://github.com/hako-mikan/sd-{w}-supermerger',               # 图像融合功能
    ]
    for url in urls:
        
        filename = url.split('/')[-1]
        
        if 'github' in url:
            get_ipython().system(f'git clone {url} {wb}/extensions/{filename}')

    get_ipython().system(f'wget https://raw.githubusercontent.com/camenduru/{sdw}-scripts/main/run_n_times.py -O {wb}/scripts/run_n_times.py')
    get_ipython().system(f'aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/embed/upscale/resolve/main/4x-UltraSharp.pth -d {wb}/models/ESRGAN -o 4x-UltraSharp.pth')

    get_ipython().system(f'wget https://huggingface.co/gmk123/wav2lip/resolve/main/wav2lip.pth -O {wb}/extensions/sd-wav2lip-uhq/scripts/wav2lip/checkpoints/wav2lip.pth')
    get_ipython().system(f'wget https://huggingface.co/gmk123/wav2lip/resolve/main/wav2lip_gan.pth -O {wb}/extensions/sd-wav2lip-uhq/scripts/wav2lip/checkpoints/wav2lip_gan.pth')
    get_ipython().system(f'wget https://huggingface.co/gmk123/wav2lip/resolve/main/s3fd-619a316812.pth -O {wb}/extensions/sd-wav2lip-uhq/scripts/wav2lip/face_detection/detection/sfd/s3fd-619a316812.pth')
    get_ipython().system(f'wget https://huggingface.co/gmk123/wav2lip/resolve/main/shape_predictor_68_face_landmarks.dat -O {wb}/extensions/sd-wav2lip-uhq/scripts/wav2lip/predicator/shape_predictor_68_face_landmarks.dat')

    # 优化embeddings
    gitDownload(f'https://huggingface.co/embed/negative',f'{wb}/embeddings/negative')
    get_ipython().system(f'rm -rf {wb}/embeddings/negative/.git')
    get_ipython().system(f'rm {wb}/embeddings/negative/.gitattributes')

    gitDownload(f'https://huggingface.co/embed/lora',f'{wb}/models/Lora/positive')
    get_ipython().system(f'rm -rf {wb}/models/Lora/positive/.git')
    get_ipython().system(f'rm {wb}/models/Lora/positive/.gitattributes')

    #中文插件
    gitDownload(f'https://github.com/DominikDoom/a1111-sd-{w}-tagcomplete',f'{wb}/extensions/a1111-sd-{w}-tagcomplete')
    get_ipython().system(f'rm -f {wb}/extensions/a1111-sd-{w}-tagcomplete/tags/danbooru.csv')
    get_ipython().system(f'wget https://beehomefile.oss-cn-beijing.aliyuncs.com/20210114/danbooru.csv -O {wb}/extensions/a1111-sd-{w}-tagcomplete/tags/danbooru.csv')
    gitDownload(f'https://github.com/toriato/{sdw}-wd14-tagger',f'{wb}/extensions/{sdw}-wd14-tagge')
    # get_ipython().system(f'rm -f {wb}/localizations')
    # gitDownload(f'https://github.com/dtlnor/{sdw}-localization-zh_CN',f'{wb}/extensions/{sdw}-localization-zh_CN')

    #附加插件=脸部修复/颜色细化/ps组件/漫画助手/分块vae
    # gitDownload(f'https://github.com/hnmr293/sd-{w}-cutoff',f'{wb}/extensions/sd-{w}-cutoff')
    gitDownload(f'https://github.com/Bing-su/adetailer',f'{wb}/extensions/adetailer')
    gitDownload(f'https://github.com/antfu/sd-{w}-qrcode-toolkit',f'{wb}/extensions/sd-{w}-qrcode-toolkit')
    gitDownload(f'https://github.com/yankooliveira/sd-{w}-photopea-embed',f'{wb}/extensions/sd-{w}-photopea-embed')
    # get_ipython().system(f'wget https://huggingface.co/gmk123/mhzs/raw/main/jubenchajian4_51.py -O {wb}/scripts/jubenchajian4_51.py')
    gitDownload(f'https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111',f'{wb}/extensions/multidiffusion-upscaler-for-automatic1111')

    # ControlNet模型
    Cnt_models = [
            'control_v11e_sd15_ip2p_fp16.safetensors',
            'control_v11e_sd15_shuffle_fp16.safetensors',
            'control_v11p_sd15_canny_fp16.safetensors',
            'control_v11f1p_sd15_depth_fp16.safetensors',
            'control_v11p_sd15_inpaint_fp16.safetensors',
            'control_v11p_sd15_lineart_fp16.safetensors',
            'control_v11p_sd15_mlsd_fp16.safetensors',
            'control_v11p_sd15_normalbae_fp16.safetensors',
            'control_v11p_sd15_openpose_fp16.safetensors',
            'control_v11p_sd15_scribble_fp16.safetensors',
            'control_v11p_sd15_seg_fp16.safetensors',
            'control_v11p_sd15_softedge_fp16.safetensors',
            'control_v11p_sd15s2_lineart_anime_fp16.safetensors',
            'control_v11f1e_sd15_tile_fp16.safetensors',
        ]
    get_ipython().system(f'rm -rf {wb}/extensions/sd-{w}-controlnet')
    # 模型下载到Colab
    if ControlNet == "Colab":
        gitDownload(f'https://github.com/Mikubill/sd-{w}-controlnet',f'{wb}/extensions/sd-{w}-controlnet')
        for v in Cnt_models:
            get_ipython().system(f'aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/{v} -d {wb}/extensions/sd-{w}-controlnet/models -o {v}')
    
    # 模型下载到Google_Drive
    elif ControlNet == "Google_Drive":
        cloudDriveSync(f'{gwb}/CntModels')
        gitDownload(f'https://github.com/Mikubill/sd-{w}-controlnet',f'{wb}/extensions/sd-{w}-controlnet')
        for v in Cnt_models:
            if not os.path.exists(f'{gwb}/CntModels/{v}'):
                get_ipython().system(f'aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/{v} -d {gwb}/CntModels -o {v}')
                print("创建扩展",f'{v}')
            else:
                print("扩展存在",f'{v}')
        # 遍历cntmodels目录下的文件，将文件银接到Colab目录下的extensions目录下
        for v in os.listdir(f'{gwb}/CntModels'):
            # 链接云盘目录
            get_ipython().system(f'ln -s {gwb}/CntModels/{v} {wb}/extensions/sd-{w}-controlnet/models')
                
    elif ControlNet == "No":
        print("不使用 ControlNet")

    # 各种UI界面
    if UI == "Kitchen_Ui":
        gitDownload(f'https://github.com/canisminor1990/sd-{w}-kitchen-theme-legacy', f'{wb}/extensions/sd-{w}-kitchen-theme-legacy')
        print("Kitchen界面插件启用")
    elif UI == "Lobe_Ui":
        gitDownload(f'https://github.com/canisminor1990/sd-web-ui-kitchen-theme', f'{wb}/extensions/sd-web-ui-kitchen-theme')      
        print("Lobe界面插件启用")
    elif UI == "No":
        print("UI插件不启用")
    
    # 关键词
    if Key_words:
        gitDownload(f'https://github.com/Physton/sd-{w}-prompt-all-in-one', f'{wb}/extensions/sd-{w}-prompt-all-in-one')
        cloudDriveSync(f'{gwb}/Storage', f'{wb}/extensions/sd-{w}-prompt-all-in-one/storage', Key_words)
        print("关键词插件启用")
    else:
        get_ipython().system(f'rm -rf {wb}/extensions/sd-{w}-prompt-all-in-one')
        print("关键词插件不启用")


# 初始化本地环境
def initLocal():
        
    #部署 env 环境变量
    get_ipython().system(f'apt -y update -qq')
    get_ipython().system(f'wget https://huggingface.co/wageguagua/sd_config/resolve/main/libtcmalloc_minimal.so.4 -O /content/libtcmalloc_minimal.so.4')
    get_ipython().run_line_magic('env', 'LD_PRELOAD=/content/libtcmalloc_minimal.so.4')

    #设置 python 环境
    get_ipython().system(f'apt -y install -qq aria2 libcairo2-dev pkg-config python3-dev')
    get_ipython().system(f'pip install -q torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 torchtext==0.15.2 torchdata==0.6.1 --extra-index-url https://download.pytorch.org/whl/cu118 -U')
    get_ipython().system(f'pip install -q xformers==0.0.20 triton==2.0.0 gradio_client==0.2.7 -U')

    #主框架模块
    if Version == "A1111":
        get_ipython().system(f'git clone -b master  https://github.com/AUTOMATIC1111/{sdw} {wb}')
    elif Version == "V2.5":
        get_ipython().system(f'git clone -b v2.5 https://github.com/camenduru/{sdw} {wb}')
    elif Version == "V2.4":
        get_ipython().system(f'git clone -b v2.4 https://github.com/camenduru/{sdw} {wb}')

    get_ipython().system(f'git -C {wb}/repositories/stable-diffusion-stability-ai reset --hard')
    
    # 初始化云盘
    initCloudDrive()

    # 安装附加功能
    installAdditional()

    # 删除原配置
    get_ipython().system(f'rm -f {wb}/config.json')
    
    # 链接用户配置
    get_ipython().system(f'ln -s {gwb}/Config/config.json {wb}/config.json')

    # 如果云盘没有模型
    if len(os.listdir(f"{gwb}/Models")) == 0:
        #下载主模型
        get_ipython().system(f'aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/wageguagua/sdmodels/resolve/main/chilloutmix_NiPrunedFp32Fix.safetensors -d {wb}/models/Stable-diffusion -o chilloutmix_NiPrunedFp32Fix.safetensors')
        get_ipython().system(f'aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/coco233/colab/resolve/main/ZR888.safetensors -d {wb}/models/Stable-diffusion -o zr888.safetensors')
    # 如果云盘Vae模型
    if len(os.listdir(f"{gwb}/Vae")) == 0:
        # #VAE
        get_ipython().system(f'aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.safetensors -d {wb}/models/VAE -o vae-ft-mse-840000-ema-pruned.safetensors')
        
    #放大
    get_ipython().system(f'aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/embed/upscale/resolve/main/4x-UltraSharp.pth -d {wb}/models/ESRGAN -o 4x-UltraSharp.pth')

    model_dir = os.path.join(wb, "models", "Stable-diffusion")
    if any(f.endswith(('.ckpt', '.safetensors')) for f in os.listdir(model_dir)):
        get_ipython().system(f'sed -i \'s@weight_load_location =.*@weight_load_location = "cuda"@\' {wb}/modules/shared.py')
        get_ipython().system(f'sed -i "s@os.path.splitext(model_file)@os.path.splitext(model_file); map_location=\'cuda\'@" {wb}/modules/sd_models.py')
        get_ipython().system(f'sed -i "s@map_location=\'cpu\'@map_location=\'cuda\'@" {wb}/modules/extras.py')
        get_ipython().system(f"sed -i 's@ui.create_ui().*@ui.create_ui();shared.demo.queue(concurrency_count=999999,status_update_rate=0.1)@' {wb}/webui.py")
        
# 运行
def run(script):
    clear_output()
    get_ipython().run_line_magic('cd', f'{wb}')
    get_ipython().system(f'python {script} --listen --xformers --enable-insecure-extension-access --theme dark --gradio-queue --disable-console-progressbars --multiple --api --cors-allow-origins=*')

# 运行脚本
if os.path.exists(f'{wb}'):
    run('webui.py')
else:
    # 初化本地环境
    initLocal()
    # 运行
    run('launch.py')
