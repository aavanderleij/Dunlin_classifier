"""
docsting blaljiojij
"""

import os
import sys

import cv2
import logging
import pandas as pd


class TrainingSetGenerator:

    def __init__(self, video, boris_file, sleap_file):
        self.foraging = ["foraging", "hunting", "hunting prey", "pecking", "pecking prey", "probing", "probing prey",
                         "sweeping", "sweeping prey", "unknown prey"]

        self.standing = ["walking", "vigilant", "interaction", "stretching", "standing still", "vomiting", "hopping"]
        self.grooming = ["scraching", "bathing", "fluttering"]

        self.video = video
        self.boris_file = boris_file

        self.sleap_df = pd.read_csv(sleap_file)

    def crop_image(self, x, y, image):
        # TODO split sort_video_frame into smaller functions for readability
        print(x)

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

        return crop_frame

    def sort_video_frames(self):

        # get file name
        pathlist = self.video.split("/")
        file_name = pathlist[-1]
        file_name = file_name.replace(".mp4", "_")
        video = cv2.VideoCapture(self.video)
        boris_df = pd.read_csv(self.boris_file, skiprows=15)
        # add frame index
        boris_df["Frame_num"] = boris_df["Time"] * boris_df["FPS"]
        # drop point events
        boris_df = boris_df[boris_df.Status != "POINT"]

        logging.info(f"video = {video}")
        success, image = video.read()
        count = 1
        # get frames form video
        while success:
            logging.debug(f"frame number = {count}")
            if count % 15 != 0:
                success, image = video.read()

                count += 1
                continue

            for i in range(0, len(boris_df.index), 2):
                logging.debug(f"{i} in {len(boris_df.index)}")
                start = int(boris_df.iloc[i]["Frame_num"])
                stop = int(boris_df.iloc[i + 1]["Frame_num"])
                behavior = str(boris_df.iloc[i]["Behavior"])
                # TODO change categories
                if behavior in self.foraging:
                    category = "foraging"
                else:
                    category = behavior

                try:
                    if start <= count <= stop:
                        logging.debug(f"frame number start = {start}")
                        logging.debug(f"frame number stop = {stop}")

                        save_loc = fr"catagories/{category}/"
                        if not os.path.exists(save_loc):
                            os.makedirs(save_loc)

                        # get sleap
                        # TODO here idx
                        frame_rows = self.sleap_df.loc[self.sleap_df["frame_idx"] == count]
                        print(frame_rows)
                        frame_rows = frame_rows.reset_index()
                        height, width, channels = image.shape
                        if height != 1080:
                            logging.error(f"video resolution {height} and {width} is not as correct")
                        logging.debug(f"height - {height}")
                        logging.debug(f"width = {width}")
                        logging.debug(f"done with frame {count}")
                        if len(frame_rows.index) == 1:

                            x = int(frame_rows["center_x"])
                            y = int(frame_rows["center_y"])

                            crop_frame = self.crop_image(x, y, image)

                            logging.info(f"{file_name}{count}.jpg in {save_loc}")
                            try:
                                cv2.imwrite(save_loc + fr"{file_name}{count}.jpg",
                                            crop_frame)  # save frame as JPEG file
                                print(cv2.imwrite(save_loc + fr"{file_name}{count}.jpg", crop_frame))
                            except Exception as e:
                                print(e)

                        else:

                            for index in frame_rows.index:
                                print("row is")
                                print(index)
                                x = int(frame_rows["center_x"][index])
                                y = int(frame_rows["center_y"][index])

                                crop_frame = self.crop_image(x, y, image)

                                logging.info(f"{file_name}{count}_{index}.jpg in {save_loc}")
                                try:
                                    cv2.imwrite(save_loc + fr"{file_name}{count}.jpg", crop_frame)
                                    print(cv2.imwrite(save_loc + fr"{file_name}{count}_{index}.jpg", crop_frame))
                                except Exception as e:
                                    print(e)

                        break
                    else:
                        continue
                except KeyError:
                    continue
            success, image = video.read()
            logging.debug(f"done with frame {count}")
            count += 1


def main():
    tg = TrainingSetGenerator(
        "/export/lv9/user/avdleij/training_data_sunny/output/rescaled/20220806_dunlin1-video1.mp4",
        "/export/lv9/user/avdleij/boris_files/20220806_dunlin1-video1.csv",
        "/export/lv9/user/avdleij/training_data_sunny/output/sleap_predictions/20220806_dunlin1"
        "-video1.slp")
    tg.sort_video_frames()


if __name__ == "__main__":
    sys.exit(main())
