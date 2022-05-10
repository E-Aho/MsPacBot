import os
import pickle
from pathlib import Path

import cv2
from ale_py import ALEInterface, roms
from src.constants import OUT_MAP

from src.Interfacing.Network import leaky_relu, linear, NeuralNetwork

from src.Genetics.Populations import Individual
from src.Genetics.GeneticsObjects import Chromosome


def create_video_from_individual(individual_chr: Chromosome, file_name: str):
    img_dir = Path(f"vids/raw/{file_name}")

    ale = ALEInterface()
    ale.setInt('random_seed', 101)
    ale.setBool('display_screen', False)
    ale.setBool('sound', False)
    ale.setString("record_screen_dir", f"vids/raw/{file_name}")
    os.mkdir(f"vids/raw/{file_name}")
    ale.loadROM(roms.MsPacman)

    input_dim = ale.getRAM().shape[0]
    output_dim = len(OUT_MAP.keys())

    architecture = [input_dim, 64, 32, output_dim]

    neural_net = NeuralNetwork(
            layer_nodes=architecture,
            hidden_activation=leaky_relu,
            output_activation=linear,
            chromosome=individual_chr,
        )

    individual = Individual(chromosome=individual_chr, network=neural_net, env=ale)
    individual.get_fitness()



def create_video(image_folder: str):

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    raw_dir = f"vids/raw/{image_folder}"

    images = [img for img in os.listdir(raw_dir) if img.endswith(".png")]
    images.sort()
    frame = cv2.imread(os.path.join(raw_dir, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(f'vids/output_{image_folder}.mp4', fourcc, 60.0, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(raw_dir, image)))

    cv2.destroyAllWindows()
    video.release()

if __name__ == "__main__":
    with open("individuals/A50T10P60Pr40", "rb") as file:
        best_chromosome = pickle.loads(file.read())
    create_video_from_individual(individual_chr=best_chromosome, file_name="best_agent")
    create_video("best_agent")
