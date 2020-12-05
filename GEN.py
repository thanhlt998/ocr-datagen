from wand.image import Image
from wand.display import display
from textwrap import wrap
from wand.color import Color
from wand.drawing import Drawing

import matplotlib.pyplot as plt
import numpy as np
from glob import glob as gl 
import imutils
from tqdm.notebook import trange
import os
import shutil
import cv2



text_config = {
    "char_list":"0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ()-*!:#",
    "word_connecter":["/", " ", " ", " ", "-", "*", ".", ",", "'"],
    "sensitive": True,
    "max_words": 5,
    "from_dict": 0.3,
    "max_length": 12,
    "duplicate": 0.05,  
}

effect_config={
    "height": 64,
    "add_line": 0.3,
    "add_circle": 0.1,
    "min_quality": 0.3,
    "blur": 0.5,
    "add_noise":0.3,
    "motion_blur": 0.5
}


def resize_image(img, min_quality):
    ratio = np.random.uniform(min_quality, 1)
    w, h, _ = img.shape
    img = cv2.resize(img, (int(h*ratio), int(w*ratio)))
    img = cv2.resize(img, (h, w))
    return img

def add_noise(img, rate):
    prop = np.random.rand()
    if prop< rate:
        gaussian_noise = np.zeros((img.shape[0], img.shape[1], 3),dtype=np.uint8)
        rd = np.random.randint(0, 100)
        cv2.randn(gaussian_noise, rd, 20)
        img = img+gaussian_noise
    return img

def gen_words(config, is_single):
    max_length = config.get("max_length")
    if is_single==1:
        max_length = 8
    text = ""
    init_random = np.random.uniform(0, 1, 2)
    if init_random[0]<config.get("from_dict"): # Get text from dictionary 
        text = words[np.random.randint(0, dic_length)]
        if config.get("sensitive") and np.random.uniform(0, 1)<0.5:
            text = text[0].title()+text[1:]
    else:
        if init_random[1]<0.25: # Text is a random integer                                               
            text = str(np.random.randint(-100000000, 10000000))
        if 0.5>init_random[1]>0.25:# Text is a real number
            stt = np.random.randint(0, 4)
            sig = np.random.rand()
            if sig<0:
                text = "-"+text
            for i in range(stt-1):
                text += str(np.random.randint(0, 1000))
                r = np.random.randint(0, 2)
                if r ==0:
                    text +="."
                else:
                    text +=","
            text += str(np.random.randint(0, 1000))
        else: # Text from random character
            char_list = config.get("char_list")
            inx = np.random.randint(0, len(char_list))
            text = str(char_list[inx])
            N = np.random.randint(-5, 15)
            if N<1:
                N=1
            inx = np.random.randint(0, len(char_list), N-1)
            for i in range(N-1):
                prop = np.random.randint(0, 100)/100
                if prop<config.get("duplicate"):
                    text+=text[-1]
                else:
                    text+=char_list[inx[i]]
    if len(text)>max_length:
        text = text[:max_length]
    return text
    
def gen_text(config):
    connecter = config.get("word_connecter")
    word_num = np.random.randint(1, config.get("max_words"))
    if word_num==1:
        text = gen_words(config, 0)
    else:
        text = gen_words(config, 1)
        for i in range(word_num-1):
            indx = np.random.randint(0, len(connecter))
            text+=connecter[indx]
            text+=gen_words(config, 1)
    r = np.random.randint(0, 100)
    if r < 5:
        text = "-"+text
    return text

def eval_metrics(image, ctx, txt):
    """Quick helper function to calculate width/height of text."""
    metrics = ctx.get_font_metrics(image, txt, True)
    return (metrics.text_width, metrics.text_height)



def draw_text(text, bg, font, fill_opacity, fill_color, text_kerning):
    font_size = 40
    with Drawing() as draw:
        draw.font = font
        draw.font_size = font_size
        draw.fill_opacity = fill_opacity
        draw.fill_color = fill_color
        draw.text_kerning = text_kerning
        draw.text_interline_spacing= np.random.uniform(0.1, 2)
        w, h = eval_metrics(gosh, draw, text)
        r = min(w, h)
        p1 = np.random.randint(0, 1.5*r)
        p2 = np.random.randint(-r//10, r//4)
        draw.text(abs(p1), abs(int(h-font_size//5)+p2), text)
        img = cv2.resize(bg, (int(w+2*p1), int(h+2*p2)))
        img = Image.from_array(img)
        draw(img)
        return img
    
def convert_out(img, min_q):
    
    img = np.array(img)
    if len(img.shape)==4:
        img =cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    return resize_image(img, min_q)
        
def apply_effects(img, effect_config):
    min_q = effect_config.get('min_quality')
    outputs = []
    outputs.append(convert_out(img, min_q))
    
    # WAY EFFECT
    loops = np.random.randint(0, 3)
    for i in range(loops):
        h_r = np.random.randint(25, 100)
        w_l = np.random.randint(1, 4)
        img.wave(amplitude=min(img.height, img.width) / h_r,
                 wave_length=img.width /  w_l)
    
    # implode EFFECT
    amount = np.random.uniform(0.01, 0.05)
    img.implode(amount=amount)
    outputs.append(convert_out(img, min_q))
    
    # BLUR EFFECT
    motion_blur_radius = np.random.randint(0, 15)
    motion_blur_sigma = np.random.randint(2, 8)
    motion_blur_angle = np.random.randint(-180, 180)
    img.motion_blur(radius=motion_blur_radius, sigma=2, angle=motion_blur_angle)
    outputs.append(convert_out(img, min_q))
    
    # NOISE EFFECT
    img = np.array(img)
    angle = np.random.randint(-5, 5)
    img = imutils.rotate_bound(img, angle)
    img =cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    img = add_noise(img, effect_config.get("add_noise"))
    outputs.append(convert_out(img, min_q))
    
    # convertScaleAbs EFFECT
    a = np.random.uniform(0.5, 1.4)
    b = np.random.randint(-30, 15)
    img = cv2.convertScaleAbs(img, alpha=a, beta=b)
    outputs.append(convert_out(img, min_q))
    
    return outputs

        


def gen_sample(text, fronts, effect_config):
    font_id = np.random.randint(0, len(fronts))
    font = fronts[font_id]
#     print(font, text)
    rand = np.random.randint(0, 100)
    if rand <50:
        bg_color = np.random.randint(200, 255, 3, np.uint8)
        bg = np.zeros((WIDTH, HEIGHT, 3), np.uint8)
        bg[:, :] = bg_color
    else:
        bg = bg_imgs[np.random.choice(len(bg_imgs))]
        
    fill_opacity = np.random.uniform(0.5, 1)
    fill_color = np.random.randint(0, 100, 3, np.uint8)
    fill_color = Color('rgb('+str(fill_color[0])+","+str(fill_color[1])+","+str(fill_color[2])+")")
    text_kerning = np.random.randint(0, 12)
    img = draw_text(text, bg, font, fill_opacity, fill_color, text_kerning)
    outputs = apply_effects(img, effect_config)
    return outputs
        

def make_dataset(samples, output, fonts, text_config, effect_config):
    try:
        shutil.rmtree(output)
    except:
        pass
    if output[-1]!="/":
        output+='/'
    try:
        os.mkdir(output)
        os.mkdir(output+"images")
    except:
        pass
    with open(output+'image_list.txt', 'a+') as the_file:
        for i in trange(samples):
            text = gen_text(text_config)
            outputs= gen_sample(text, fonts_file, effect_config)
            file_name = str(i)
#             plt.imshow(outputs[0])
#             plt.show()
#             plt.imshow(outputs[3])
#             plt.show()
#             plt.imshow(outputs[4])
#             plt.show()
            text = text.replace("-", "@")
            the_file.write("images/"+file_name+".jpg"+ "\t"+ text+'\n')
            the_file.write("images/"+file_name+"&.jpg"+ "\t"+ text+'\n')
            the_file.write("images/"+file_name+"&&.jpg"+ "\t"+ text+'\n')
            the_file.write("images/"+file_name+"&&&.jpg"+ "\t"+ text+'\n')
            the_file.write("images/"+file_name+"&&&&.jpg"+ "\t"+ text+'\n')
            
            file = output+ "images/"+file_name+".jpg"
            file1 = output+ "images/"+file_name+"&.jpg"
            file2 = output+ "images/"+file_name+"&&.jpg"
            file3 = output+ "images/"+file_name+"&&&.jpg"
            file4 = output+ "images/"+file_name+"&&&&.jpg"
            
            cv2.imwrite(file, outputs[0])
            cv2.imwrite(file1, outputs[1])
            cv2.imwrite(file2, outputs[2])
            cv2.imwrite(file3, outputs[3])
            cv2.imwrite(file4, outputs[4])
        the_file.close()


        
    
HEIGHT = effect_config.get("height")
WIDTH=HEIGHT*4
gosh = np.ones((256, 64, 3), np.uint8)*255
gosh = Image.from_array(gosh)

words = np.load("words.npy")[:1000]
dic_length = len(words)
output_path = "out/"
fonts_file = gl("all_fonts/*.ttf")
bg_imgs = []
bg_files = gl("backgrounds/*.jpg")
print("num_bg:", len(bg_files))
print("num fronts:", len(fonts_file))
for i in range(len(bg_files)):
    img = cv2.imread(bg_files[i], 1)
    bg_imgs.append(img)

output = "out/"
