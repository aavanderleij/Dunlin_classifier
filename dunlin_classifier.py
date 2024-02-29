#!/usr/bin/env python3
# TODO make stand alone exe
"""
The Dunlin Classifier was developed for generating behaviour time-series from videos featuring dunlins.
The program produces a csv file containing predictions, prediction scores, and pixel coordinates for dunlins found in
frames.

Autor: Antsje van der Leij
"""
import logging

logging.basicConfig(level=logging.WARNING)

import csv
# imports
import os
import sys
import argparse
import numpy as np
import shutil

import subprocess
import pandas as pd
import cv2
import tensorflow as tf

from dunlinclassifier.SLEAP_model import SLEAPModel
from dunlinclassifier.SLEAP_parser import SleapParser


class DunlinClassifier:

    def __init__(self):
        args = self.get_args()
        self.video_dir = os.path.join(args.video_dir, "")
        self.frame_step = args.frame_step

        if args.output_dir:
            self.output_dir = args.output_dir
        else:
            self.output_dir = os.path.join(self.video_dir, "output/")

        # log argparse values
        logging.info(f"argparse video dir = {self.video_dir}")
        logging.info(f"argparse frame_step = {self.frame_step}")
        logging.info(f"argparse output_dir = {self.output_dir}")

        self.rescale_output = os.path.join(self.output_dir, "rescaled/")
        self.temp_dir = os.path.join(self.output_dir, "tmp/")
        self.sleap_prediction_path = os.path.join(self.output_dir, "sleap_predictions/")

        self.result_dict = dict()

    def get_args(self, argv=None):
        # argparse
        parser = argparse.ArgumentParser(
            prog='dunlin classifier',
            description='A classifier for dunlin behavior. Trained for horizontal side view videos of dunlin on the '
                        'waddensea.')
        parser.add_argument('video_dir', help='path to the directory containing the videos to be tracked', type=str)
        parser.add_argument('-o', '--output_dir', help='path to the directory to store the csv output files',
                            type=str, default='')
        parser.add_argument('-f', '--frame_step', help='only predict every n-th frame. Will reduce prediction time if'
                                                       ' set to higher value. Set value to 1 to ',
                            type=int,
                            default=15)

        return parser.parse_args(argv)

    def get_files_from_dir(self, path, file_extension):
        """
        Gets the filenames of all files in a directory if it ends with the given extension.

        :param path: str
        :param file_extension: str
        :return: list of file names
        """
        print("getting files")

        # check if file exists
        if not os.path.isdir(path):
            sys.exit("File path to videos does not exist or is incorrect!")

        # get only one type of file
        files = [f for f in os.listdir(path) if f.endswith(file_extension) or f.endswith(file_extension.upper())]

        # check if any files match file type
        if not files:
            logging.error("no " + file_extension + " found in " + path)
            sys.exit("no " + file_extension + " found in " + path)

        return files

    def rescale_video(self, video_name):
        """
        Saves a copy of the input video at the standard size of 1920X1080p
        Does not upscale and adds black bars to keep original aspect ratio.

        :param video_name:
        :return:
        """
        # path to video
        video = self.video_dir + video_name

        logging.info(f"rescale videos are saved at: {self.rescale_output}")

        # if output dir doesn't exist, create it
        if not os.path.isdir(self.rescale_output):
            os.makedirs(self.rescale_output)
            logging.debug("made a new directory for rescaled videos")

        # save location and name of rescaled video
        ffmpeg_output = self.rescale_output + video_name

        # ffmpeg command for rescaling a video to 1980p and adding black bars to retain original aspect ratio
        command = [
            "ffmpeg",
            "-i",
            video,
            "-vf",
            "scale=1920:1080:force_original_aspect_ratio=decrease:eval=frame,pad=1920:1080:-1:-1:color=black",
            ffmpeg_output
        ]

        # runs command
        subprocess.run(command)

    def run_sleap(self):
        """
        initiates a SLEAP model and runs interferes over all videos in the directory

        :return:
        """
        sleap_model = SLEAPModel(self.rescale_output)
        print(self.rescale_output)
        # run sleap inference on video
        sleap_model.predict(self.output_dir)

    def get_sleap_df(self):
        """
        changes parses .slp and saves output to a csv file
        :return:
        """
        sleap_parser = SleapParser()
        sleap_prediction_path = os.path.join(self.output_dir, "sleap_predictions/")
        for file in self.get_files_from_dir(sleap_prediction_path, ".slp"):
            if not os.path.isdir(self.output_dir + "sleap_csv/"):
                os.makedirs(os.path.join(self.output_dir, "sleap_csv/"))
            df = sleap_parser.sleap_to_pandas(sleap_prediction_path + file)
            df.to_csv(os.path.join(self.output_dir, "sleap_csv/", file.replace(".slp", ".csv")))

    # TODO function is way to long, needs work
    def crop_frame(self, video_name):
        """
        creates small images of dunlin by getting the pixel coordinates from the sleap model and then cropping
        the image around said coordinates.
        :param video_name: string containing the video name
        :return:
        """

        print(f"isolating dunlin from frames for {video_name}...")

        # add video as key for results dict
        self.result_dict[video_name] = {}
        # TODO feedback to user how many frame done/to go

        # get file name
        video = self.rescale_output + video_name
        video_id = video_name.replace(".mp4", "")

        # get sleap dataframe
        sleap_df = pd.read_csv(os.path.join(self.output_dir, "sleap_csv/", video_id + ".csv"))

        cropped_img_path = os.path.join(self.output_dir, "cropped_frames/", video_id + "/")
        # make dir for saving image
        if not os.path.exists(cropped_img_path):
            os.makedirs(cropped_img_path)
        else:
            shutil.rmtree(cropped_img_path)
            os.makedirs(cropped_img_path)

        # open video
        video = cv2.VideoCapture(video)
        # get total amount of frames
        total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
        print(f"{video_id} has {total_frames} frames")

        # check if first frame can be opened
        success, image = video.read()
        count = 1

        # get frames from video as long as the next frame can be accessed
        while success:

            logging.debug(f"frame number = {count}")
            logging.debug(f"frame step = {self.frame_step}")
            logging.debug(count % self.frame_step)

            # crop frames in a set interval
            if count % self.frame_step != 0:
                count += 1
                # don't do anything and got to the next frame

            else:
                # prepare dict
                frame_dict = {}
                self.result_dict[video_name][count] = frame_dict

                # get SLEAP data for this video frame
                frame_rows = sleap_df.loc[sleap_df["frame_idx"] == count]

                # logging
                logging.debug("frame_rows =")
                logging.debug(frame_rows)

                # check if SLEAP found an instance
                if frame_rows.empty:
                    frame_dict["sleap_hit"] = False
                else:
                    frame_dict["sleap_hit"] = True

                # reset index
                frame_rows = frame_rows.reset_index()
                height, width, channels = image.shape
                logging.debug(f"height - {height}")
                logging.debug(f"width = {width}")

                # if only one instance is found in this frame the result will be a data array, not a data frame
                if len(frame_rows.index) == 1:

                    # prepare dict
                    instance_dict = {}

                    # get prediction score
                    sleap_score = frame_rows["score"]
                    sleap_score = sleap_score.tolist()[0]
                    instance_dict["sleap_score"] = sleap_score

                    # get coordinates for center point of bird
                    x = int(frame_rows["center_x"])
                    y = int(frame_rows["center_y"])

                    # save coordinates in results
                    instance_dict["coordinate_bird_x"] = x
                    instance_dict["coordinate_bird_y"] = y

                    # set corner points for crop around bird
                    x_start = x - 90
                    x_end = x + 90

                    y_start = y - 70
                    y_end = y + 70

                    # logging
                    logging.info(f"center coordinates = x: {x} and y: {y}")
                    logging.debug("crop frame borders points")
                    logging.debug(f"x start = {x_start}")
                    logging.debug(f"y start = {y_start}")
                    logging.debug(f"x end = {x_end}")
                    logging.debug(f"y end = {y_end}")

                    # crop frame around dunlin
                    crop_frame = image[y_start:y_end, x_start: x_end]
                    logging.info(f"frame{count}.jpg in {self.output_dir} + cropped_frames/ + {video_id}")

                    try:
                        # save cropped frame as jpeg
                        write_success = cv2.imwrite(
                            self.output_dir + "cropped_frames/" + video_id + "/" + fr"frame_{count}.jpg",
                            crop_frame)
                        instance_dict["frame_crop_succes"] = write_success
                    except Exception as e:
                        print(e)

                    # save results in dict
                    frame_dict[0] = instance_dict

                # TODO lots of duplicate code
                # if more than 1 instance is found, treat as dataframe
                else:
                    for index in frame_rows.index:

                        instance_dict = {}
                        # get center coordinates from dataframe
                        x = int(frame_rows["center_x"][index])
                        y = int(frame_rows["center_y"][index])

                        # save coordinates in results
                        instance_dict["coordinate_bird_x"] = x
                        instance_dict["coordinate_bird_y"] = y

                        # save sleap score
                        sleap_score = frame_rows["score"]
                        sleap_score = sleap_score.tolist()[0]
                        instance_dict["sleap_score"] = sleap_score

                        x_start = x - 90
                        x_end = x + 90

                        y_start = y - 70
                        y_end = y + 70

                        # crop frame around bird
                        crop_frame = image[y_start:y_end, x_start: x_end]
                        logging.info(f"frame{count}.jpg in {self.output_dir} + cropped_frames/ + {video_id}")
                        try:
                            # save croped frame as jpeg
                            write_success = cv2.imwrite(self.output_dir + "cropped_frames/" + video_id + "/" +
                                                        fr"frame_{count}_{index}.jpg", crop_frame)
                            instance_dict["frame_crop_succes"] = write_success
                        except Exception as e:
                            logging.warning("crop frame borders sizes")
                            logging.warning(f"x start = {x_start}")
                            logging.warning(f"y start = {y_start}")
                            logging.warning(f"x end = {x_end}")
                            logging.warning(f"y end = {y_end}")
                            print(e)
                        # add sleap score to result dict
                        frame_dict[index] = instance_dict
                count += 1

            # logging
            logging.debug(f"done with frame {count}")
            logging.debug(f"image {count} out of {total_frames}")
            logging.debug(f"succes = {success}")
            # check next frame
            success, image = video.read()

    def run_keras(self, video):
        """
        runs a keras image classification model and save the results in a dictionary
        :param video: str
        :return:
        """

        # classes
        class_names = ["empty", "foraging", "non_foraging"]

        # load model
        model = tf.keras.models.load_model('models/Keras_model/3class_dunlin.keras')
        print(model.summary())

        # get video ID
        video_id = video.replace(".mp4", "")
        labeled_predictions_list = []
        prediction_score_list = []

        video_dict = self.result_dict[video]

        # file were img to predict are stored
        img_dir = os.path.join(self.output_dir, "cropped_frames/" + video_id)

        for img in self.get_files_from_dir(img_dir, ".jpg"):

            # load img
            loaded_img = tf.keras.utils.load_img(img_dir + "/" + img, target_size=(140, 180))

            # convert to array for prediction
            img_array = tf.keras.utils.img_to_array(loaded_img)
            img_array = tf.expand_dims(img_array, 0)

            # predict
            pred = model.predict(img_array)

            # get best scoring label
            labeled_pred = class_names[np.argmax(pred[0], axis=-1)]
            # save prediction in list
            labeled_predictions_list.append(labeled_pred)

            # get score of best scoring label
            score = np.max(tf.nn.softmax(pred[0]))
            # save score in list
            prediction_score_list.append(score)

            # get file name for cropped img
            img = img.replace(".jpg", "")
            img_split = img.split("_")
            frame_idx = img_split[1]
            if len(img_split) == 3:
                instance_num = int(img_split[-1].split(".")[0])
            else:
                instance_num = 0

            logging.debug(f"frame idx = {frame_idx}")

            # this triggers a key error sometimes, but I don't know why
            # only happens sometimes when instance_num >= 1
            # HPC issue?
            try:
                video_dict[int(frame_idx)][instance_num]["label"] = labeled_pred
                video_dict[int(frame_idx)][instance_num]["classification_score"] = score

            except KeyError:
                logging.warning(f"KeyError at {img}")

            # TODO add option to remove cropped frames after classification

    def write_results(self):
        """
        parses the result dictionary and saves it as a CSV file
        :return:
        """

        header = ["video", "frame", "instance", "sleap_score", "coordinate_bird_x",
                  "coordinate_bird_y", "label", "classification_score"]

        with open(self.output_dir + "output.csv", "wt") as fp:
            writer = csv.writer(fp, delimiter=",")
            writer.writerow(header)

            logging.debug(self.result_dict)

            for video in self.result_dict:
                video_dict = self.result_dict[video]
                for frame in video_dict:
                    sleap_hit = video_dict[frame].pop("sleap_hit")
                    if sleap_hit:
                        for instance in video_dict[frame]:
                            instance_dict = video_dict[frame][instance]
                            sleap_score = instance_dict.pop("sleap_score")
                            coordinate_bird_x = instance_dict.pop("coordinate_bird_x")
                            coordinate_bird_y = instance_dict.pop("coordinate_bird_y")
                            if instance_dict:
                                try:

                                    label = instance_dict["label"]
                                    classification_score = instance_dict["classification_score"]
                                    row = [video, frame, instance, sleap_score, coordinate_bird_x,
                                           coordinate_bird_y, label, classification_score]
                                    writer.writerow(row)
                                except KeyError:
                                    print("label error")
                            else:
                                label = "no_hit"
                                row = [video, frame, instance, sleap_score, coordinate_bird_x,
                                       coordinate_bird_y, label, None]
                                writer.writerow(row)
                    else:
                        row = [video, frame, None, None, None,
                               None, "no_hit", None]
                        writer.writerow(row)

    def annotate_videos(self):
        """
        complete pipline from video to CSV output
        :return:
        """
        # get all videos from input dir
        video_list = self.get_files_from_dir(self.video_dir, ".mp4")
        for video in video_list:
            print(video)
            # rescale video
            self.rescale_video(video)

        # run sleap
        print("running sleap")
        self.run_sleap()
        self.get_sleap_df()

        for video in video_list:
            print(video)
            self.crop_frame(video)
            self.run_keras(video)
        print("writing results...")
        self.write_results()


# TODO clean up all temp files


# classes


# main
def main():
    print("starting classifier...")
    # parse command line arguments
    dc = DunlinClassifier()
    dc.annotate_videos()
    # dc.training_conf_mat()


if __name__ == "__main__":
    sys.exit(main())
