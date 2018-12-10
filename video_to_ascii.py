import cv2
import time
import sys
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import string
import moviepy.editor as mp
import multiprocessing
from multiprocessing.pool import ThreadPool

def convert_txt_to_png(ascii_string, height, width):
    #print("Converting txt to png")
    img = np.zeros((width, height, 3))
    offset_y = 0
    offset_x = 0
    for i, char in enumerate(ascii_string):
        if char == '\n':
            offset_y += 1
            offset_x = 0
        else:
            char_matrix = ascii_dict[char]
            img[offset_y*size_of_char[0]:offset_y*size_of_char[0]+size_of_char[0],
                offset_x*size_of_char[1]:offset_x*size_of_char[1]+size_of_char[1],
                :] += char_matrix
            offset_x += 1
    img_save = Image.fromarray(np.uint8(img))
    return img_save

def grey_to_ascii(grey):
    ascii_list = [' ', '.', ':', '-', '=', '+', '*', '#', '&', '%', '@']
    return ascii_list[int(grey/25)]

def img_to_string(array):
    ascii_string = ''
    chars_fit_h = int(array.shape[0]/size_of_char[0])
    chars_fit_w = int((array.shape[1]/size_of_char[1]))
    chunks = []
    for i in range(chars_fit_h):
        for j in range(chars_fit_w):
            chunks.append(array[i*size_of_char[0]:i*size_of_char[0] + size_of_char[0],
                                j*size_of_char[1]:j*size_of_char[1] + size_of_char[1]])
        chunks.append(np.empty((0,0)))
    #print(len(chunks))
    #print(chars_fit_h*chars_fit_w)
    #print(chars_fit_h, chars_fit_w)
    for chunk in chunks:
        if chunk.shape != size_of_char:
            pass
        if chunk.size == 0:
            ascii_string += '\n'
        else:
            avg = np.average(chunk)
            char = grey_to_ascii(avg)
            ascii_string += char
    return ascii_string

def Asciimap():
    text = string.printable
    text = text.replace('\n', '')
    fnt = ImageFont.truetype(font, font_size)
    ascent, descent = fnt.getmetrics()
    size_of_char = fnt.getsize('0')
    im = Image.new("RGB", (len(text)*size_of_char[0], size_of_char[1]))
    draw = ImageDraw.Draw(im)
    draw.text((0, 0), text, font=fnt, fill=rgb)
    im = im.crop(im.getbbox())
    im.save("fonts/bitmap.png")
    ascii_dict = {}
    im = np.array(im)
    if im.shape[0] != size_of_char[0] or im.shape[1]/len(text) != size_of_char[1]:  # This detects problem with font
        print("Detected a size problem with your font at the given size, adjusting pixel values")
        print(f"Size_x should be {im.shape[0]} but font loader detected size as {size_of_char[0]}")
        print(f"Size_y should be {int(im.shape[1]/len(text))} but font loader detected size as {size_of_char[1]}")
        size_of_char = (im.shape[0], im.shape[0])#int(im.shape[1]/len(text)))
    q = [im[:, i*size_of_char[1]:i*size_of_char[1]+size_of_char[1], :] for i in range(len(text))]
    """
    for i in range(len(text)):
        cv2.imshow('img', q[i])
        cv2.waitKey(0)
    """
    for i, char in enumerate(text):
        if q[i].shape != (size_of_char[0], size_of_char[1], 3):
            print(f"Character {char} did not fit in bitmap.")
            print(f"is {q[i].shape} but should be {(size_of_char[0], size_of_char[1], 3)}")
        else:
            ascii_dict[char] = q[i]
            """
            print(char)
            char_matrix = cv2.resize(q[i], dsize=(100, 100), interpolation=cv2.INTER_CUBIC)
            cv2.imshow('img',  char_matrix)
            cv2.waitKey(0)
            """
    print(ascii_dict.keys())
    return ascii_dict, size_of_char

def render_frame(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #img = cv2.imread("IMG_2673.JPG", 0)
    #height, width = img.shape  # No channel because grey image
    img = cv2.resize(img, dsize=(height, width), interpolation=cv2.INTER_CUBIC)
    ascii_string = img_to_string(np.array(img))
    """
    with open("img.txt", 'w') as file:
        file.write(ascii_string)
    """
    #print(ascii_string)
    txt_img = convert_txt_to_png(ascii_string, height, width)
    txt_img = cv2.cvtColor(np.array(txt_img), cv2.COLOR_BGR2RGB)  # Convert to rgb
    txt_img = np.array(txt_img)
    return txt_img

def process_init(*args):
    global width, height, font_size, font, size_of_char, rgb, ascii_dict, is_video
    width = args[0]
    height = args[1]
    font_size = args[2]
    font = args[3]
    size_of_char = args[4]
    rgb = args[5]
    ascii_dict = args[6]
    is_video = args[7]

if __name__ == "__main__":
    # Change stuff here, enable
    width, height = 800, 1422  # Height, Width
    font_size = 4  # Some fucky stuff happens at 3, so don't set it to 3.
    rgb = (0, 0, 255)  # RGB values of ascii
    is_video = True  # Set to False if you want to use picture instead
    pic_path = "Examples/Zergling.jpg"  # Name of pic file
    pic_output = "results/ascii_image"  # Output name pic
    vid_path = "Examples/20181124_135911.mp4"  # Name of video file
    vid_output = "results/video"  # Output name video
    processes = 10  # How many processes to use when rendering video

    font = 'fonts/zig_monospace.ttf'
    ascii_dict, size_of_char = Asciimap()
    print(f"Initiaizing {processes} processes")
    p = multiprocessing.Pool(processes, initializer=process_init, initargs=(width, height, font_size, font, size_of_char, rgb, ascii_dict, is_video))
    print("Initialization completed")
    if is_video:
        cap = cv2.VideoCapture(vid_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        amount_of_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        print("FRAMES = ", amount_of_frames)
        print("FPS = ", fps)
        fourcc = cv2.VideoWriter_fourcc(*"DIVX")
        video = cv2.VideoWriter(vid_output + ".mp4", fourcc, fps, (height, width))  # FPS is equal to source
    else:
        cap = cv2.imread(pic_path)
        amount_of_frames = 1


    grabbed_frame = True
    img_number = 0
    total_time = 0
    print("Starting render")
    while grabbed_frame:
        start_time = time.time()  # ETA
        frames_to_process = []
        for f in range(processes):
            if is_video:
                grabbed_frame, frame = cap.read()
                if grabbed_frame:
                    frames_to_process.append(frame)
            else:
                frame = cap
                frames_to_process.append(frame)

        processed_frames = p.map(render_frame, frames_to_process)
        for processed_frame in processed_frames:
            if is_video:
                video.write(processed_frame)
            else:
                processed_frame = Image.fromarray(np.uint8(processed_frame))
                processed_frame.save(pic_output + ".png")
                processed_frame = np.array(processed_frame)
                grabbed_frame = False

            cv2.imshow('img', processed_frame)
            cv2.waitKey(1)
        img_number += processes
        elapsed_time = time.time() - start_time
        total_time += elapsed_time  # ETA
        eta = round((total_time/img_number)*(amount_of_frames-img_number))
        print(f'{round(img_number/amount_of_frames*100)}% done, eta {eta} seconds   ', end='\r')

    if is_video:
        video.release()
        cap.release()
        cv2.destroyAllWindows()

        # Sync audio to video
        video = mp.VideoFileClip(vid_path)
        video.audio.write_audiofile(vid_output + ".mp3")
        video.close()

        # Open ascii video and add audio
        video = mp.VideoFileClip(vid_output + ".mp4")
        audio = mp.AudioFileClip(vid_output + ".mp3")
        video = video.set_audio(audio)
        video.write_videofile(vid_output + "_audio.mp4", fps=fps)  # NEED TO SET IT TO OTHER NAME, OR ELSE IT WILL SUPERBUG
        #video.write_videofile(vid_output + "_audio.webm", fps=fps)
        video.close()
        audio.close()
        #video.write_gif(vid_output + ".gif", fps=fps)