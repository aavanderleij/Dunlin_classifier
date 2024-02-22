#!/usr/bin/env python
"""
Uses a model that was trained with SLEAP and uses it to estimate keypoint positions of the red knot in video's.

autor: Antsje van der Leij

"""

import sys
import os
import logging
import glob
import subprocess
import sleap
from dunlin_classifier.SLEAP_parser import SleapParser


class SLEAPModel:
    """
    A Class that predicts animal poses using a video as input.
    Returns a csv file with the pixel coordinates of key points found bij the model.
    Uses a model that is trained using the SLEAP GUI.
    """

    def __init__(self, video_dir, predictions_out_dir="predictions/"):

        self.video_dir = video_dir
        self.model = self.load_model()
        self.predictions_out_dir = predictions_out_dir

    def get_files_from_dir(self, path, file_extension):
        """
        gets files names ending with the file exetntion from target directory and returns those files as a list
        :param path: absolute path to target directory
        :param file_extension: the extension of retrieved files
        :return: list with files ending with the file_extension
        """

        # check if file exists
        if not os.path.isdir(path):
            sys.exit("File path to videos does not exist or is incorrect!")
            print(path)

        # get only one type of file
        files = [f for f in os.listdir(path) if f.endswith(file_extension) or f.endswith(file_extension.upper())]

        # check if any files match file type
        if not files:
            sys.exit("no " + file_extension + " found in " + path)

        return files

    def load_video(self, path_to_video):
        """
        Loads a mp4 video as SLEAP video object
        :param path_to_video:
        :return: SLEAP video object
        """
        print("load video")

        loaded_video = sleap.load_video(path_to_video)
        return loaded_video

    def load_model(self):
        """
        loads a trained SLEAP model directory model
        """
        print("load model")

        # use glob to make a variable model path so name of model doesn't matter
        model_path = glob.glob('models/SLEAP_models/*')
        logging.debug(f"model loaded from: {model_path}")

        model = sleap.load_model(model_path)
        logging.info("sleap model loaded")

        return model

    def run_model(self, video):
        """
        Loads a pre-trained model from SLEAP.
        Runs the model on video to generate predictions.
        Predictions are then saved.
        :param video: video file name
        :return: SLEAP Labels object
        """

        print('running model...')

        # run model

        labels = self.model.predict(video)
        labels = sleap.Labels(labels.labeled_frames)

        return labels

    def predict(self, output_dir):
        """
        run model over every video in target directory and actives tracking if tracking is True
        :param instance_count: (int) amount of expected animals in videos
        :param tracking: (boolean)

        """

        videos = self.get_files_from_dir(self.video_dir, ".mp4")
        print(videos)
        # TODO add x out of x video
        for video in videos:

            print("run prediction for:")
            print(video)
            # use video name as name for predictions save file
            save_name = video.replace(".mp4", "")
            save_name = save_name.replace(".MP4", "")
            sleap_video = self.load_video(self.video_dir + "/" + video)

            # file path to save sleap predictions
            sleap_pred_dir = output_dir + "sleap_predictions/"
            slp_file = sleap_pred_dir + save_name + ".slp"

            # make directory for sleap predictions one doesn't exist
            if not os.path.isdir(sleap_pred_dir):
                os.makedirs(sleap_pred_dir)
                logging.debug("made a new directory for sleap predictions")

            # most common error is KeyError while indexing videos
            try:
                labels = self.run_model(sleap_video)
                labels.save(slp_file)
                logging.info(f"slp file at {slp_file}")
            # ffmpeg command is a quick fix for KeyError while indexing
            except KeyError:
                print("ran into error while indexing video: " + video)
                print("Attempting to fix it. please wait...")
                subprocess.run(
                    ["ffmpeg", "-y", "-i", self.video_dir + "/" + video, "-c:v", "libx264", "-pix_fmt", "yuv420p",
                     "-preset", "superfast", "-crf", "23", self.video_dir + "/fixed" + video])
                try:
                    sleap_video = self.load_video(self.video_dir + "/fixed" + video)
                    labels = self.run_model(sleap_video)
                    labels.save(slp_file)
                    logging.info(f"slp file at {slp_file}")
                except KeyError:
                    print("unable to fix video")
                    print("continue with next video (if there are any)")
                    continue





def main():
    print("in main")

    model = SLEAPModel(args.video_dir)
    model.predict()


if __name__ == "__main__":
    sys.exit(main())
