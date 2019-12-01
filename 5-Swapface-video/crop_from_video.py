import cv2
import os
import argparse
from tqdm import tqdm
import dlib

def get_face(img, detector):
    dets = detector(img, 1)
    if dets:
        left = dets[0].left()
        top = dets[0].top()
        right = dets[0].right()
        bot = dets[0].bottom()

        return True, img[top:bot, left:right]
    return False, None

def process_video(video_path, save_path, detector, num_faces):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 1000)
    
#     pbar = tqdm(total=cap.get(cv2.CAP_PROP_FRAME_COUNT), desc = 'frames_process')
    pbar_faces = tqdm(total=num_faces, desc = 'faces_process')
    n = 0
    while cap.isOpened() and n < num_faces:
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        save_image = os.path.join(save_path, str(n) +'.bmp')
        
        inf, img_face = get_face(frame, detector)
        if inf:
            img_face = cv2.cvtColor(img_face, cv2.COLOR_BGR2RGB)
            cv2.imwrite(save_image, img_face)
            pbar_faces.update()
            n += 1
            
#         pbar.update()
#         if ret==True:
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
                
#     pbar.close()
    pbar_faces.close()
    cap.release()
#     cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CropFaces')
    parser.add_argument('--video_path_a', type=str, default='video/putin.mp4',
                        help='path to video of face A')
    parser.add_argument('--video_path_b', type=str, default='video/trump.mp4',
                        help='path to video of face B')
    parser.add_argument('--save_faces_path_a', type=str, default='train/face_A',
                        help='path to save faces A')
    parser.add_argument('--save_faces_path_b', type=str, default='train/face_B',
                        help='path to save faces B')
    parser.add_argument('--num_faces', type=int, default=500, metavar='N',
                        help='number of faces for both')

    args = parser.parse_args()
    print(args)

    detector = dlib.get_frontal_face_detector()

    if not os.path.exists(args.save_faces_path_a):
        os.makedirs(args.save_faces_path_a)

    if not os.path.exists(args.save_faces_path_b):
        os.makedirs(args.save_faces_path_b)

    process_video(args.video_path_a, args.save_faces_path_a, detector, args.num_faces)
    process_video(args.video_path_b, args.save_faces_path_b, detector, args.num_faces)
