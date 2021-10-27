from __future__ import print_function
import os
import csv
import pandas as pd
import argparse
import dlib

import face_recognition
from face_recognition import face_locations

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--save_feature', type=str,
                        default='feature.csv',
                        help='path to the feature file to be save (default: feature.csv)')

    parser.add_argument('--read_csv', type=str,
                        default='Fairlabel.csv',
                        help='path to the csv file to read (default: Fairlabel.csv)')
    args = parser.parse_args()

    print("extracting face features from images")
    feature_vecs = []
    fnames = []
    final_fnames = []
    failed_fnames = []

    with open(args.read_csv) as file:
    	reader = csv.reader(file)
    	next(reader)
    	for line in reader:
    		fnames.append(line[0])

    for fname in fnames:
        img_path = fname

        # face detection
        print(img_path)
        X_img = face_recognition.load_image_file(img_path)
        X_faces_loc = face_locations(X_img)
        # if the number of faces detected in a image is not 1, ignore the image
        if len(X_faces_loc) == 0:
            failed_fnames.append(fname)
            continue
        # extract 128 dimensional face features
        final_fnames.append(fname)
        faces_encoding = dlib.face_recognition_model_v1(X_img, known_face_locations=X_faces_loc)[0]
        feature_vecs.append(faces_encoding)

    df_feat = pd.DataFrame(feature_vecs, index=final_fnames)
    df_feat.sort_index(inplace=True)
    df_feat.to_csv(args.save_feature)
    print(failed_fnames)


if __name__ == "__main__":
    main()
