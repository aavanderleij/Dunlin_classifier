#!/usr/bin/env python
# TODO change shebang/make stand alone exe
"""
the goal is to take a video of a dunlin in the wild and classify its behavior
"""
import csv
# imports
import os
import sys
import argparse
import numpy as np
import logging
import subprocess

import pandas as pd
import cv2
import tensorflow as tf
from keras.preprocessing import image

from dunlin_classifier.SLEAP_model import SLEAPModel
from dunlin_classifier.SLEAP_parser import SleapParser

logging.basicConfig(level=logging.INFO)


class DunlinClassifier:

    def __init__(self):
        args = self.get_args()
        self.video_dir = args.video_dir
        self.frame_step = args.frame_step
        # TODO remove hard coded
        self.boris_files = "/export/lv9/user/avdleij/boris_files/"

        if args.output_dir:
            self.output_dir = args.output_dir
        else:
            self.output_dir = self.video_dir + "output/"

        # log argparse values
        logging.info(f"argparse video dir = {self.video_dir}")
        logging.info(f"argparse frame_step = {self.frame_step}")
        logging.info(f"argparse output_dir = {self.output_dir}")

        self.rescale_output = self.output_dir + "rescaled/"
        self.temp_dir = self.output_dir + "tmp/"
        self.sleap_prediction_path = self.output_dir + "sleap_predictions/"

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
        parser.add_argument('-f', '--frame_step', help='only predict every n-th frame. Will reduce prediction time.',
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
        print("get files")

        # check if file exists
        if not os.path.isdir(path):
            print(path)
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
        https://superuser.com/questions/547296/resizing-videos-with-ffmpeg-avconv-to-fit-into-static-sized-player/1136305#1136305
        https://trac.ffmpeg.org/wiki/Scaling

        :param video_name:
        :param output_dir:
        :return:
        """
        # path to video
        video = self.video_dir + video_name

        logging.info(f"rescale videos are saved at: {self.rescale_output}")

        # if output dir doesn't exist, create it
        if not os.path.isdir(self.temp_dir):
            os.makedirs(self.temp_dir)
            logging.debug("made a new directory for rescaled videos")

        # save location and name of rescaled video
        ffmpeg_output = self.temp_dir + video_name

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

    def reindex_video(self, video_name):
        """
        Runs a ffmpeg comand to reindex vidoes that have been rescaled.
        This prevents index errors and shifting coordinates.
        :param video_name:
        :return:
        """

        # path to video
        video = self.temp_dir + video_name

        logging.info(f"reindx videos are saved at: {self.rescale_output}")

        if not os.path.isdir(self.rescale_output):
            os.makedirs(self.rescale_output)
            logging.debug("made a new directory for rescaled videos")

        # save location and name of rescaled video
        ffmpeg_output = self.rescale_output + video_name
        print("ran into error while indexing video: " + video)
        print("Attempting to fix it. please wait...")
        subprocess.run(
            ["ffmpeg", "-y", "-i", video, "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-preset", "superfast", "-crf", "23", ffmpeg_output])


    def create_conda_environment(self, env_name, requirements_file):
        """
        activates a conda env and if it doesn't exist, creates it
        :param env_name:
        :param requirements_file:
        :return:
        """
        # TODO check if still needed
        # Example usage:
        # create_conda_environment("env1", "requirements.txt")
        env_exists = False
        try:
            subprocess.run(f"conda activate {env_name} ", shell=True, check=True)
            env_exists = True
        except subprocess.CalledProcessError as e:
            pass

        if not env_exists:
            subprocess.run(f"conda create --name {env_name} --file {requirements_file}", shell=True)
            print(f"Conda environment {env_name} created.")
        else:
            print(f"Conda environment {env_name} already exists.")

    def run_sleap(self):
        """
        initiates a SLEAP model and runs interferes over all videos in the directory
        :return:
        """
        sleap_model = SLEAPModel(self.rescale_output)
        # run sleap inference on video
        sleap_model.predict(self.output_dir)

    def get_sleap_df(self):
        """
        changes parses .slp and saves output to a csv file
        :return:
        """
        sleap_parser = SleapParser()
        sleap_prediction_path = self.output_dir + "sleap_predictions/"
        for file in self.get_files_from_dir(sleap_prediction_path, ".slp"):
            if not os.path.isdir(self.output_dir + "sleap_csv/"):
                os.makedirs(self.output_dir + "sleap_csv/")
            df = sleap_parser.sleap_to_pandas(sleap_prediction_path + file)
            df.to_csv(self.output_dir + "sleap_csv/" + file.replace(".slp", ".csv"))

    def crop_frame(self, video_name):

        # add video as key for results dict
        self.result_dict[video_name] = {}
        # make list to track frames

        # get file name
        video = self.rescale_output + video_name

        video_id = video_name.replace(".mp4", "")

        # get sleap dataframe
        sleap_df = pd.read_csv(self.output_dir + "sleap_csv/" + video_id + ".csv")

        # make dir for saving image
        if not os.path.exists(self.output_dir + "cropped_frames/" + video_id + "/"):
            os.makedirs(self.output_dir + "cropped_frames/" + video_id + "/")

        video = cv2.VideoCapture(video)
        logging.info(f"video = {video}")
        success, image = video.read()
        count = 1
        # get frames form video
        while success:
            logging.debug(f"frame number = {count}")
            if count % self.frame_step != 0:
                success, image = video.read()

                count += 1
                continue
            else:
                frame_dict = {}
                self.result_dict[video_name][count] = frame_dict

                # get sleap data for this video frame
                frame_rows = sleap_df.loc[sleap_df["frame_idx"] == count]
                # check if sleap model found a positive
                if frame_rows.empty:
                    frame_dict["sleap_hit"] = False
                else:
                    frame_dict["sleap_hit"] = True

                frame_rows = frame_rows.reset_index()
                height, width, channels = image.shape
                logging.debug(f"height - {height}")
                logging.debug(f"width = {width}")
                logging.debug(f"done with frame {count}")

                if len(frame_rows.index) == 1:

                    index = 0
                    instance_dict = {}

                    # save prediction score
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

                    logging.info(f"center coordinalts = x: {x} and y: {y}")

                    logging.debug("crop frame borders sizes")
                    logging.debug(f"x start = {x_start}")
                    logging.debug(f"y start = {y_start}")
                    logging.debug(f"x end = {x_end}")
                    logging.debug(f"y end = {y_end}")

                    # crop frame around bird
                    crop_frame = image[y_start:y_end, x_start: x_end]
                    logging.info(f"frame{count}.jpg in {self.output_dir} + cropped_frames/ + {video_id}")
                    try:
                        # save croped frame as jpeg
                        write_success = cv2.imwrite(
                            self.output_dir + "cropped_frames/" + video_id + "/" + fr"frame_{count}.jpg",
                            crop_frame)
                        instance_dict["frame_crop_succes"] = write_success
                    except Exception as e:
                        print(e)

                    frame_dict[index] = instance_dict

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
                            logging.error("crop frame borders sizes")
                            logging.error(f"x start = {x_start}")
                            logging.error(f"y start = {y_start}")
                            logging.error(f"x end = {x_end}")
                            logging.error(f"y end = {y_end}")
                            print(e)
                        # add sleap score to result dict
                        frame_dict[index] = instance_dict
                count += 1

    def run_keras(self, video):

        # classes
        class_names = ["foraging", "no bird", "open wings", "self care", "sleeping", "unknown", "vigilant"]

        # load model
        model = tf.keras.models.load_model('dunlin_classifier/my_model.keras')
        print(model.summary())

        video_id = video.replace(".mp4", "")
        labeled_predictions_list = []
        prediction_score_list = []


        video_dict = self.result_dict[video]

        # file were img to predict are stored
        img_dir = self.output_dir + "cropped_frames/" + video_id

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

            img = img.replace(".jpg", "")
            img_split = img.split("_")
            frame_idx = img_split[1]
            if len(img_split) == 3:
                instance_num = int(img_split[-1].split(".")[0])
            else:
                instance_num = 0


            video_dict[int(frame_idx)][instance_num]["label"] = labeled_pred
            video_dict[int(frame_idx)][instance_num]["classification_score"] = score

            # TODO remove croped frames after classification

    def write_results(self):

        header = ["video", "frame", "instance", "sleap_score", "coordinate_bird_x",
                  "coordinate_bird_y", "label", "classification_score"]

        with open(self.output_dir + "output.csv", "wt") as fp:
            writer = csv.writer(fp, delimiter=",")
            writer.writerow(header)  # write header

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
                                label = instance_dict["label"]
                                classification_score = instance_dict["classification_score"]
                                row = [video, frame, instance, sleap_score, coordinate_bird_x,
                                       coordinate_bird_y, label, classification_score]
                                writer.writerow(row)
                            else:
                                label = "no_hit"
                                row = [video, frame, instance, sleap_score, coordinate_bird_x,
                                       coordinate_bird_y, label, None]
                                writer.writerow(row)
                    else:
                        row = [video, frame, None, None, None,
                               None, "no_hit", None]
                        writer.writerow(row)

    def classify_dunlin(self):
        # get all videos from input dir
        video_list = self.get_files_from_dir(self.video_dir, ".mp4")
        for video in video_list:
            # rescale video
            self.rescale_video(video)
            # reindex
            self.reindex_video(video)
            # for every video run sleap
        self.run_sleap()
        self.get_sleap_df()

        for video in video_list:
            self.crop_frame(video)
            self.run_keras(video)
        print("writhing results")
        self.write_results()

    def make_more_training_data(self):
        video_list = self.get_files_from_dir(self.video_dir, ".mp4")
        for video in video_list:
            # rescale video
            self.rescale_video(video)
            # reindex
            self.reindex_video(video)
            # for every video run sleap
        self.run_sleap()
        self.get_sleap_df()

        for video in video_list:
            self.crop_frame(video)
            self.model_assisted_labeling(video)

    def model_assisted_labeling(self, video):
        # TODO (re)move

        # classes
        class_names = ["foraging", "no bird", "open wings", "self care", "sleeping", "unknown", "vigilant"]

        # load model
        model = tf.keras.models.load_model('dunlin_classifier/my_model.keras')
        print(model.summary())

        video_id = video.replace(".mp4", "")

        video_dict = self.result_dict[video]
        print(video_dict)
        # file were img to predict are stored
        img_dir = self.output_dir + "cropped_frames/" + video_id

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

            # get score of best scoring label
            score = np.max(tf.nn.softmax(pred[0]))

            # make dir for saving image
            if not os.path.exists(self.output_dir + "model_assisted_labeling/"):
                os.makedirs(self.output_dir + "model_assisted_labeling/")

            for label in class_names:
                if not os.path.exists(self.output_dir + "model_assisted_labeling/" + label + "/"):
                    os.makedirs(self.output_dir + "model_assisted_labeling/" + label + "/")

            open_img = cv2.imread(img_dir + "/" + img)

            cv2.imwrite(self.output_dir + "model_assisted_labeling/" + labeled_pred + "/" + video_id + "_" + img, open_img)

    def training_conf_mat(self):
        print("making matrix")
        true_labels = []
        pred_labels = []
        from sklearn.metrics import confusion_matrix

        # classes
        class_names = ["foraging", "no bird", "open wings", "self care", "sleeping", "unknown", "vigilant"]

        # load model
        model = tf.keras.models.load_model('dunlin_classifier/my_model.keras')
        print(model.summary())

        train_data = "/export/lv9/user/avdleij/currated_croped_training_img/catagories/sorted_labels/"
        for name in class_names:
            img_dir = train_data + name
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

                true_labels.append(name)
                pred_labels.append(labeled_pred)

        result = confusion_matrix(true_labels, pred_labels, normalize='pred', labels=class_names)
        print(result)
        df_cm = pd.DataFrame(result, class_names, class_names)

        df_cm.to_csv("train_conf_mat.csv")









    # TODO clean up all temp files

    # constants

    # functions


# classes


# main
def main():
    print("starting classifier...")
    # parse command line arguments
    dc = DunlinClassifier()
    # dc.classify_dunlin()

    dc.training_conf_mat()


if __name__ == "__main__":
    sys.exit(main())
