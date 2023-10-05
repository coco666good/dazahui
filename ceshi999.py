import argparse #line:1
import binascii #line:2
def str_to_bool (O000O0O00OOOO0O0O ):#line:5
    if O000O0O00OOOO0O0O .lower ()in ('yes','true','t','y','1'):#line:6
        return True #line:7
    elif O000O0O00OOOO0O0O .lower ()in ('no','false','f','n','0'):#line:8
        return False #line:9
    else :#line:10
        raise argparse .ArgumentTypeError ('Invalid boolean value: {}'.format (O000O0O00OOOO0O0O ))#line:11
parser =argparse .ArgumentParser ()#line:14
parser .add_argument ('--UI',type =str )#line:16
parser .add_argument ('--Version',type =str )#line:17
parser .add_argument ('--ControlNet',type =str_to_bool )#line:18
args =parser .parse_args ()#line:21
UI =args .UI #line:23
Version =args .Version #line:24
ControlNet =args .ControlNet #line:25
import sys #line:29
import os #line:30
import base64 #line:31
import importlib .util #line:32
from IPython import get_ipython #line:33
from IPython .display import clear_output #line:34
import tensorflow as tf #line:35
print ("TensorFlow version:",tf .__version__ )#line:37
if tf .test .gpu_device_name ():#line:38
    print ("GPU is available")#line:39
else :#line:40
    print ("GPU is NOT available")#line:41
    raise Exception ("\n没有使用GPU，请在代码执行程序-更改运行时类型-设置为GPU！\n如果不能使用GPU，建议更换账号！")#line:42
w =base64 .b64decode (("d2VidWk=").encode ('ascii')).decode ('ascii')#line:44
sdw =base64 .b64decode (("c3RhYmxlLWRpZmZ1c2lvbi13ZWJ1aQ==").encode ('ascii')).decode ('ascii')#line:45
wb =f'/content/guagua'#line:48
get_ipython ().run_line_magic ('cd','/content')#line:50
get_ipython ().run_line_magic ('env','TF_CPP_MIN_LOG_LEVEL=1')#line:51
def gitDownload (O0OOOO0000O000000 ,OOO0O0O00OOOOOO00 ):#line:54
    if os .path .exists (OOO0O0O00OOOOOO00 ):#line:55
        return #line:56
    get_ipython ().system (f'git clone {O0OOOO0000O000000} {OOO0O0O00OOOOOO00}')#line:58
def installAdditional ():#line:62
    O0000O000O0000O0O =[f'https://github.com/camenduru/{sdw}-images-browser',f'https://github.com/camenduru/sd-{w}-tunnels',f'https://github.com/etherealxx/batchlinks-{w}',f'https://github.com/camenduru/sd-civitai-browser',f'https://github.com/AUTOMATIC1111/{sdw}-rembg',f'https://github.com/thomasasfk/sd-{w}-aspect-ratio-helper',f'https://github.com/hanamizuki-ai/{sdw}-localization-zh_Hans',f'https://github.com/kohya-ss/sd-{w}-additional-networks',f'https://github.com/fkunn1326/openpose-editor',f'https://github.com/hnmr293/posex',f'https://github.com/nonnonstop/sd-{w}-3d-open-pose-editor',f'https://github.com/camenduru/{sdw}-catppuccin',f'https://github.com/KohakuBlueleaf/a1111-sd-{w}-lycoris',f'https://github.com/Physton/sd-{w}-prompt-all-in-one',]#line:83
    for O00000OOO00OO0OOO in O0000O000O0000O0O :#line:84
        O000O0OO0OOOO0O00 =O00000OOO00OO0OOO .split ('/')[-1 ]#line:86
        if 'github'in O00000OOO00OO0OOO :#line:88
            get_ipython ().system (f'git clone {O00000OOO00OO0OOO} {wb}/extensions/{O000O0OO0OOOO0O00}')#line:89
    get_ipython ().system (f'rm -rf {wb}/embeddings/negative')#line:91
    gitDownload (f'https://huggingface.co/embed/negative',f'{wb}/embeddings/negative')#line:92
    get_ipython ().system (f'rm -rf {wb}/embeddings/negative/.git')#line:93
    get_ipython ().system (f'rm {wb}/embeddings/negative/.gitattributes')#line:94
    gitDownload (f'https://github.com/DominikDoom/a1111-sd-{w}-tagcomplete',f'{wb}/extensions/a1111-sd-{w}-tagcomplete')#line:97
    get_ipython ().system (f'rm -f {wb}/extensions/a1111-sd-{w}-tagcomplete/tags/danbooru.csv')#line:98
    get_ipython ().system (f'wget https://beehomefile.oss-cn-beijing.aliyuncs.com/20210114/danbooru.csv -O {wb}/extensions/a1111-sd-{w}-tagcomplete/tags/danbooru.csv')#line:99
    gitDownload (f'https://github.com/toriato/{sdw}-wd14-tagger',f'{wb}/extensions/{sdw}-wd14-tagge')#line:100
    gitDownload (f'https://github.com/Bing-su/adetailer',f'{wb}/extensions/adetailer')#line:102
    O0000OO0O0O0OOO0O =['control_v11e_sd15_ip2p_fp16.safetensors','control_v11e_sd15_shuffle_fp16.safetensors','control_v11p_sd15_canny_fp16.safetensors','control_v11f1p_sd15_depth_fp16.safetensors','control_v11p_sd15_inpaint_fp16.safetensors','control_v11p_sd15_lineart_fp16.safetensors','control_v11p_sd15_mlsd_fp16.safetensors','control_v11p_sd15_normalbae_fp16.safetensors','control_v11p_sd15_openpose_fp16.safetensors','control_v11p_sd15_scribble_fp16.safetensors','control_v11p_sd15_seg_fp16.safetensors','control_v11p_sd15_softedge_fp16.safetensors','control_v11p_sd15s2_lineart_anime_fp16.safetensors','control_v11f1e_sd15_tile_fp16.safetensors',]#line:120
    get_ipython ().system (f'rm -rf {wb}/extensions/sd-{w}-controlnet')#line:121
    if ControlNet :#line:123
        gitDownload (f'https://github.com/Mikubill/sd-{w}-controlnet',f'{wb}/extensions/sd-{w}-controlnet')#line:124
        for OOO00O00OO00000O0 in O0000OO0O0O0OOO0O :#line:125
            get_ipython ().system (f'aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/{OOO00O00OO00000O0} -d {wb}/extensions/sd-{w}-controlnet/models -o {OOO00O00OO00000O0}')#line:126
        print ("启用 ControlNet")#line:127
    else :#line:128
        print ("不启用 ControlNet")#line:129
    if UI =="Kitchen_Ui":#line:132
        gitDownload (f'https://github.com/canisminor1990/sd-{w}-kitchen-theme-legacy',f'{wb}/extensions/sd-{w}-kitchen-theme-legacy')#line:133
        print ("Kitchen界面插件启用")#line:134
    elif UI =="Lobe_Ui":#line:135
        gitDownload (f'https://github.com/canisminor1990/sd-web-ui-kitchen-theme',f'{wb}/extensions/sd-web-ui-kitchen-theme')#line:136
        print ("Lobe界面插件启用")#line:137
    elif UI =="No":#line:138
        print ("UI插件不启用")#line:139
def initLocal ():#line:142
    get_ipython ().system (f'apt -y update -qq')#line:144
    get_ipython ().system (f'wget https://huggingface.co/wageguagua/sd_config/resolve/main/libtcmalloc_minimal.so.4 -O /content/libtcmalloc_minimal.so.4')#line:145
    get_ipython ().run_line_magic ('env','LD_PRELOAD=/content/libtcmalloc_minimal.so.4')#line:146
    get_ipython ().system (f'apt -y install -qq aria2 libcairo2-dev pkg-config python3-dev')#line:149
    get_ipython ().system (f'pip install -q torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 torchtext==0.15.2 torchdata==0.6.1 --extra-index-url https://download.pytorch.org/whl/cu118 -U')#line:150
    get_ipython ().system (f'pip install -q xformers==0.0.20 triton==2.0.0 gradio_client==0.2.7 -U')#line:151
    if Version =="V2.5":#line:154
        get_ipython ().system (f'git clone -b v2.5 https://github.com/camenduru/{sdw} {wb}')#line:155
    elif Version =="V2.4":#line:156
        get_ipython ().system (f'git clone -b v2.4 https://github.com/camenduru/{sdw} {wb}')#line:157
    get_ipython ().system (f'git -C {wb}/repositories/stable-diffusion-stability-ai reset --hard')#line:159
    installAdditional ()#line:162
    get_ipython ().system (f'wget -O {wb}/config.json "https://huggingface.co/wageguagua/main/raw/main/config.json"')#line:164
    get_ipython ().system (f'aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/wageguagua/sdmodels/resolve/main/long9.safetensors -d {wb}/models/Stable-diffusion -o long9.safetensors')#line:167
    get_ipython ().system (f'aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.safetensors -d {wb}/models/VAE -o vae-ft-mse-840000-ema-pruned.safetensors')#line:170
    get_ipython ().system (f'aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/embed/upscale/resolve/main/4x-UltraSharp.pth -d {wb}/models/ESRGAN -o 4x-UltraSharp.pth')#line:173
    O00O000OOO0000O00 =os .path .join (wb ,"models","Stable-diffusion")#line:175
    if any (O0OOOO00O0OOOO0OO .endswith (('.ckpt','.safetensors'))for O0OOOO00O0OOOO0OO in os .listdir (O00O000OOO0000O00 )):#line:176
        get_ipython ().system (f'sed -i \'s@weight_load_location =.*@weight_load_location = "cuda"@\' {wb}/modules/shared.py')#line:177
        get_ipython ().system (f'sed -i "s@os.path.splitext(model_file)@os.path.splitext(model_file); map_location=\'cuda\'@" {wb}/modules/sd_models.py')#line:178
        get_ipython ().system (f'sed -i "s@map_location=\'cpu\'@map_location=\'cuda\'@" {wb}/modules/extras.py')#line:179
        get_ipython ().system (f"sed -i 's@ui.create_ui().*@ui.create_ui();shared.demo.queue(concurrency_count=999999,status_update_rate=0.1)@' {wb}/webui.py")#line:180
def run (OOOO000OOO0000O0O ):#line:184
    clear_output ()#line:185
    get_ipython ().run_line_magic ('cd',f'{wb}')#line:186
    get_ipython ().system (f'python {OOOO000OOO0000O0O} --listen --enable-insecure-extension-access --theme dark --gradio-queue --multiple --opt-sdp-attention --cors-allow-origins=*')#line:187
if os .path .exists (f'{wb}'):#line:190
    run ('webui.py')#line:191
else :#line:192
    initLocal ()#line:194
    run ('launch.py')