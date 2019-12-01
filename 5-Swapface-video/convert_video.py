import cv2
import argparse
import torch
import os
import dlib
from tqdm import tqdm
from models import Autoencoder, toTensor, var_to_np
from image_augmentation import random_warp
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def extract_face(frame):
    detector = dlib.get_frontal_face_detector()
    img = frame
    dets = detector(img, 1)
    for idx, face in enumerate(dets):
        position = {}
        position['left'] = face.left()
        position['top'] = face.top()
        position['right'] = face.right()
        position['bot'] = face.bottom()
        croped_face = img[position['top']:position['bot'], position['left']:position['right']]

        return position, croped_face

def convert_face(croped_face):
    resized_face = cv2.resize(croped_face, (256, 256))
    normalized_face = resized_face / 255.0
    
    warped_img, _ = random_warp(normalized_face)
    batch_warped_img = np.expand_dims(warped_img, axis=0)

    batch_warped_img = toTensor(batch_warped_img)
    batch_warped_img = batch_warped_img.to(device).float()

    model = Autoencoder().to(device)
    checkpoint = torch.load('./checkpoint/autoencoder.t7')
    model.load_state_dict(checkpoint['state'])

    converted_face = model(batch_warped_img,'B')
    return converted_face

def merge(postion, face, body):
    mask = 255 * np.ones(face.shape, face.dtype)
    width, height, channels = body.shape
    center = (postion['left']+(postion['right']-postion['left'])//2, postion['top']+(postion['bot']-postion['top'])//2)
    normal_clone = cv2.seamlessClone(face, body, mask, center, cv2.NORMAL_CLONE)
    return normal_clone

def process_video(path_in, path_out, st_frame, num_frames):
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    cap_in = cv2.VideoCapture(path_in)
    cap_out = cv2.VideoWriter(path_out, fourcc, 3, (1280, 720))
    
    cap_in.set(cv2.CAP_PROP_POS_FRAMES, st_frame)
    n = 0
    while cap_in.isOpened() and n < num_frames:
        print(n)
        _, frame = cap_in.read()

        position, croped_face= extract_face(frame)
        
        converted_face = convert_face(croped_face)
        converted_face = converted_face.squeeze(0).detach().cpu().numpy()
        converted_face = converted_face.transpose(1,2,0)
        converted_face = np.clip(converted_face * 255, 0, 255).astype('uint8')

#         back_size = cv2.resize(converted_face, (croped_face.shape[0]-120, croped_face.shape[1]-120))
        back_size = cv2.resize(converted_face, (croped_face.shape[0], croped_face.shape[1]))

        merged = merge(position, back_size, frame)
        cap_out.write(merged)
        n += 1
    cap_out.release()
    cap_in.release()

if __name__ == '__main__':
    process_video('video\\putin.mp4', 'video\\processed.avi', 2000, 500)