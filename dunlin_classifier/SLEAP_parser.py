#!/usr/bin/env python
"""
extracts information from slp files such as node coordinates.

"""
import os
import sys

import numpy
import pandas
import sleap


class SleapParser:
    """
    Parses the information out of the .slp files generated by SLEAP_model.py and saves the data as a csv file
    """

    def get_files_from_dir(self, path, file_extension):
        """
        gets all files from a directory that with given file extension
        :param path: string containing absolute path to directory
        :param file_extension: string containing file extension
        :return:
        """

        print("get files")

        # check if file exists
        if not os.path.isdir(path):
            print(path)
            sys.exit("File path to videos does not exist or is incorrect! path must be a directory!")

        # get only one type of file
        files = [f for f in os.listdir(path) if f.endswith(file_extension) or f.endswith(file_extension.upper())]

        # check if any files match file type
        if not files:
            sys.exit("no " + file_extension + " found in " + path)

        return files

    def create_dataframe(self, node_names):
        """
        creates an empty pandas dataframe for sleap_to_pandas to fill
        :param node_names: list with strings
        :return: empty pandas dataframe
        """

        print("making data frame...")
        # make list with standard SLEAP data
        sleap_video_datatypes = [("video", str), ("frame_idx", int), ("instance_id", int), ("score", int)]
        # add nodes from the skeleton used in de video
        for name in node_names:
            # set x and y coordinates
            name_x = name + "_x"
            name_y = name + "_y"
            sleap_video_datatypes.append((name_x, int))
            sleap_video_datatypes.append((name_y, int))

        data_types = numpy.dtype(sleap_video_datatypes)
        # make empty pandas data frame
        data_frame = pandas.DataFrame(numpy.empty(0, dtype=data_types))
        return data_frame

    def sleap_to_pandas(self, filename):
        """
        converters a SLEAP file (.slp) to a pandas data frame
        :param filename:
        :return:
        """
        # load SLEAP (.slp) file
        labels = sleap.load_file(filename)
        # get node names from skeleton
        node_names = labels.skeleton.node_names
        # make empty data frame
        data_frame = self.create_dataframe(node_names)
        # get video name for dataframe csv
        file_path = labels.video.backend.filename
        video_name = file_path.split("/")
        video_name = video_name[-1]
        video_name = video_name.replace(".mp4", "")
        print("SLEAP to Pandas...")
        for frame in labels.labeled_frames:
            frame_id = frame.frame_idx
            for count, instance in enumerate(frame.predicted_instances):
                # print(instance.skeleton.node_names)
                node_points = instance.nodes_points
                score = instance.score
                # make a new row for data frame
                new_row = {"video": file_path, "frame_idx": frame_id, "instance_id": count, "score": score}
                for point in node_points:
                    # get name of node
                    node_name = point[0].name
                    # get PredictedPoint object
                    p_point = point[1]
                    # get coordinates form predictedPoint object
                    tuple_point = tuple(p_point)
                    x, y = tuple_point[:2]
                    node_name_x = node_name + "_x"
                    node_name_y = node_name + "_y"
                    # save coordinates in new_row
                    new_row[node_name_x] = [x]
                    new_row[node_name_y] = [y]
                df1 = pandas.DataFrame(new_row)
                # TODO change to dict or list. pandas concat is slow
                data_frame = pandas.concat([data_frame, df1], ignore_index=True)
        # change extension for file name
        return data_frame

    def save_as_csv(self, predict_dir, output_dir):
        print("converting to csv")

        files = self.get_files_from_dir(predict_dir, ".slp")
        print(files)
        for file in files:
            self.sleap_to_pandas(predict_dir + "/" + file, output_dir)

        print("output can be found at:")
        print(output_dir)


def main():
    ...


if __name__ == "__main__":
    sys.exit(main())
