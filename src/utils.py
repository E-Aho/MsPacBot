import os
import cv2

video_name = "vid.avi"


def create_video(image_folder: str):

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort()
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter('output2.mp4', fourcc, 60.0, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()

