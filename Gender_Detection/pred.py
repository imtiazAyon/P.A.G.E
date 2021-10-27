from __future__ import print_function
import argparse
import os
import face_recognition
import numpy as np
import sklearn
import pickle
from face_recognition import face_locations
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import cv2
import pandas as pd
import csv

COLS = ['Gender', 'Asian', 'White', 'Black', 'Indian', 'Others', 'Child', 'Adolescent', 'Teen', 'Young Adult', 'Adult', 'Old Adult', 'Senior']
N_UPSCLAE = 1
def extract_features(img_path):
    """Exctract 128 dimensional features
    """
    X_img = face_recognition.load_image_file(img_path)
    locs = face_locations(X_img, number_of_times_to_upsample = N_UPSCLAE)
    if len(locs) == 0:
        return None, None
    face_encodings = face_recognition.face_encodings(X_img, known_face_locations=locs)
    return face_encodings, locs

def predict_one_image(img_path, clf, labels):
    """Predict face attributes for all detected faces in one image
    """
    face_encodings, locs = extract_features(img_path)
    if not face_encodings:
        return None, None
    pred = pd.DataFrame(clf.predict_proba(face_encodings),
                        columns = labels)
    pred = pred.loc[:, COLS]
    return pred, locs
def draw_attributes(img_path, df):
    """Write predicted face attributes on the image
    """
    races =['A', 'W', 'B', 'I', 'O']
    ages =['C', 'Adst', 'T', 'Y Ad', 'Ad', 'Old Ad', 'Se']
    img = cv2.imread(img_path)
    # img  = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    data = {
      'Face':0,
      'Male':0,
      'Female':0,
      'A':0,
      'W':0,
      'B':0,
      'I':0,
      'O':0,
      'C':0,
      'Adst':0,
      'T':0,
      'Y Ad':0,
      'Ad':0,
      'Old Ad':0,
      'Se':0
    }
    for row in df.iterrows():
        top, right, bottom, left = row[1][13:].astype(int)
        data['Face'] = data['Face'] + 1
        if row[1]['Gender'] <= 0.5:
            gender = 'M'
            data['Male'] = data['Male'] + 1
        else:
            gender = 'F'
            data['Female'] = data['Female'] + 1
        race = races[int(np.argmax(row[1][1:6]))]
        age = ages[int(np.argmax(row[1][6:13]))]
        data[race] = data[race] + 1
        data[age] = data[age] + 1
        text_showed = f"{age} {race} {gender}"
        # No face boundary now
        #cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2) 
        font = cv2.FONT_HERSHEY_DUPLEX
        img_width = img.shape[1]
        font_scale = cv2.getFontScaleFromHeight(font, img.shape[1], 1)
        cv2.putText(img, text_showed, (left + 6, bottom - 6), font, 0.5, (0, 0, 0), 2)
        cv2.putText(img, text_showed, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
    data['img'] = img
    return data



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str,
                        default='test/',
                        help='input image directory (default: test/)')
    parser.add_argument('--output_dir', type=str,
                        default='results/',
                        help='output directory to save the results (default: results/')
    parser.add_argument('--model', type=str,
                        default='face_model_Fair_6.pkl',
                        help='path to trained model (default: face_model_Fair_6.pkl)')

    args = parser.parse_args()
    output_dir = args.output_dir
    input_dir = args.img_dir
    model_path = args.model

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    # load the model
    with open(model_path, 'rb') as f:
        clf, labels = pickle.load(f,encoding='latin1')

    print("classifying images in {}".format(input_dir))
    data_list = []
    for fname in tqdm(os.listdir(input_dir)):
        img_path = os.path.join(input_dir, fname)
        try:
            pred, locs = predict_one_image(img_path, clf, labels)
        except:
            print("Skipping {}".format(img_path))
        if not locs:
            continue
        locs = \
            pd.DataFrame(locs, columns = ['top', 'right', 'bottom', 'left'])
        df = pd.concat([pred, locs], axis=1)
        data = draw_attributes(img_path, df)
        img = data['img']
        data['img'] = fname
        data_list.append(data)
        cv2.imwrite(os.path.join(output_dir, fname), img)
        os.path.splitext(fname)[0]
        output_csvpath = os.path.join(output_dir,
                                      os.path.splitext(fname)[0] + '.csv')
        df.to_csv(output_csvpath, index = False)
    with open(os.path.join(output_dir,'output.csv'), 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['img','Face','Male','Female','A','W','B','I','O','C','Adst','T','Y Ad','Ad','Old Ad','Se'] )
        writer.writeheader()
        for dic in data_list:
            writer.writerow(dic)
    return 0

if __name__ == "__main__":
    main()

