# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 19:25:38 2021

@author: Ha V. Pham
"""
import os
import shutil
from sklearn.model_selection import train_test_split
from PIL import Image, ImageEnhance
from PIL.ImageFilter import (BLUR, SHARPEN)
import numpy as np
import pandas as pd

path = os.path.dirname(os.path.realpath(__file__))
#print(path)
classes = ["stop", "yield", "pedestrianCrossing", "speedLimit15", "speedLimit25", "speedLimit30", "speedLimit35", "speedLimit40", "speedLimit45", "speedLimit50", "speedLimit55","speedLimit60", "speedLimit65"]
df = pd.read_csv("allAnnotations.csv")
df.drop(df.columns[1], axis = 1, inplace = True)
df = df.rename(columns = {df.columns[0]: "information"})
df = pd.DataFrame(df.information.str.split(';').tolist(), columns = ['path','type', 'leftX', 'leftY', 'rightX', 'rightY', 'ocluded'])
df.drop(df.columns[6], axis = 1, inplace = True)
df.type = df.type.apply(lambda y: np.nan if y not in classes else y)
df = df.dropna()
#print(df)

X = df[['path', 'leftX', 'leftY', 'rightX', 'rightY']]
Y = df[['type']]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)
train = pd.concat([X_train, y_train], axis=1)
test = pd.concat([X_test, y_test], axis=1)
train.to_csv("train.csv")
test.to_csv("test.csv")

trainpath = "train"
if not os.path.exists(trainpath): #create a new folder for new images if not exists
    os.makedirs(trainpath)
testpath = "test"
if not os.path.exists(testpath): #create a new folder for new images if not exists
    os.makedirs(testpath)
    
for index, row in train.iterrows():
    filename = row["path"].split("/")[-1].replace(".png", ".jpg")
    newpath = trainpath + "/"+ filename
    #print("###", os.listdir(path + "/train"))
    with Image.open(row["path"]) as im:
        img_width, img_height = im.size
    obj_width = int(row["rightX"]) - int(row["leftX"])
    obj_height = int(row["rightY"]) - int(row["leftY"])
    obj_mid_x = (int(row["rightX"]) + int(row["leftX"])) / 2.0
    obj_mid_y = (int(row["rightY"]) + int(row["leftY"])) / 2.0

    obj_width_rel = obj_width / img_width
    obj_height_rel = obj_height / img_height
    obj_mid_x_rel = obj_mid_x / img_width
    obj_mid_y_rel = obj_mid_y / img_height
    if row["type"] == "stop":
        index = "0"
    elif row["type"] == "yield":
        index = "1"
        with Image.open(row["path"]) as im:
            enhancer = ImageEnhance.Brightness(im)
            dark = enhancer.enhance(0.6)
            bright = enhancer.enhance(1.4)
            dark.save("train/dark_"+ filename)
            bright.save("train/bright_"+ filename)

    elif row["type"] == "pedestrianCrossing":
        index = "2"
    else:
        index = "3"
    if filename in os.listdir(path + "/train"):
        with open(path + "/train/" + filename.replace(".jpg", ".txt"), "a") as img_txt:
            img_txt.write(index + " " + " " + str(obj_mid_x_rel) + " " + str(obj_mid_y_rel) + " " +  str(obj_width_rel) + " " + str(obj_height_rel) + "\n")
        if index == "1" and "dark_" + filename.replace(".jpg", ".txt") not in os.listdir(path + "/train"):
            with open("train.txt", "a") as train_txt:
                train_txt.write("data/obj/train/dark_" + filename + "\n")
                train_txt.write("data/obj/train/bright_" + filename + "\n")
            shutil.copy(path + "/train/" + filename.replace(".jpg", ".txt"), path + "/train/dark_" + filename.replace(".jpg", ".txt"))
            shutil.copy(path + "/train/" + filename.replace(".jpg", ".txt"), path + "/train/bright_" + filename.replace(".jpg", ".txt"))
                
    elif filename in os.listdir(path + "/test"):
        with open(path + "/test/" + filename.replace(".jpg", ".txt"), "a") as img_txt:
            img_txt.write(index + " " + " " + str(obj_mid_x_rel) + " " + str(obj_mid_y_rel) + " " +  str(obj_width_rel) + " " + str(obj_height_rel) + "\n")
        shutil.copy(path + "/test/" + filename.replace(".jpg", ".txt"), path + "/train/" + filename.replace(".jpg", ".txt"))
        if index == "1" and "dark_" + filename.replace(".jpg", ".txt") not in os.listdir(path + "/train"):
            shutil.copy(path + "/test/" + filename.replace(".jpg", ".txt"), path + "/train/dark_" + filename.replace(".jpg", ".txt"))
            shutil.copy(path + "/test/" + filename.replace(".jpg", ".txt"), path + "/train/bright_" + filename.replace(".jpg", ".txt"))
            with open("train.txt", "a") as train_txt:
                train_txt.write("data/obj/train/dark_" + filename + "\n")
                train_txt.write("data/obj/train/bright_" + filename + "\n")
        with open("train.txt", "a") as train_txt:
            train_txt.write("data/obj/train/" + filename + "\n")    
    else:
        if index == "1":  
            with open(path + "/train/dark_" + filename.replace(".jpg", ".txt"), "w+") as img_txt:
                img_txt.write(index + " " + " " + str(obj_mid_x_rel) + " " + str(obj_mid_y_rel) + " " +  str(obj_width_rel) + " " + str(obj_height_rel) + "\n")
            with open(path + "/train/bright_" + filename.replace(".jpg", ".txt"), "w+") as img_txt:
                img_txt.write(index + " " + " " + str(obj_mid_x_rel) + " " + str(obj_mid_y_rel) + " " +  str(obj_width_rel) + " " + str(obj_height_rel) + "\n")
            with open("train.txt", "a") as train_txt:
                train_txt.write("data/obj/train/dark_" + filename + "\n")
                train_txt.write("data/obj/train/bright_" + filename + "\n")
        with open(path + "/train/" + filename.replace(".jpg", ".txt"), "w+") as img_txt:
            img_txt.write(index + " " + " " + str(obj_mid_x_rel) + " " + str(obj_mid_y_rel) + " " +  str(obj_width_rel) + " " + str(obj_height_rel) + "\n")
        with open("train.txt", "a") as train_txt:
            train_txt.write("data/obj/train/" + filename + "\n")
    shutil.copy(row["path"], newpath)
        

for index, row in test.iterrows():
    filename = row["path"].split("/")[-1].replace(".png", ".jpg")
    newpath = testpath + "/"+ filename
    #print("###", os.listdir(path + "/train"))
    with Image.open(row["path"]) as im:
        img_width, img_height = im.size
    obj_width = int(row["rightX"]) - int(row["leftX"])
    obj_height = int(row["rightY"]) - int(row["leftY"])
    obj_mid_x = (int(row["rightX"]) + int(row["leftX"])) / 2.0
    obj_mid_y = (int(row["rightY"]) + int(row["leftY"])) / 2.0

    obj_width_rel = obj_width / img_width
    obj_height_rel = obj_height / img_height
    obj_mid_x_rel = obj_mid_x / img_width
    obj_mid_y_rel = obj_mid_y / img_height
    if row["type"] == "stop":
        index = "0"
    elif row["type"] == "yield":
        index = "1"
        with Image.open(row["path"]) as im:
            enhancer = ImageEnhance.Brightness(im)
            dark = enhancer.enhance(0.6)
            bright = enhancer.enhance(1.4)
            dark.save("test/dark_"+ filename)
            bright.save("test/bright_"+ filename)

    elif row["type"] == "pedestrianCrossing":
        index = "2"
    else:
        index = "3"
    if filename in os.listdir(path + "/test"):
        with open(path + "/test/" + filename.replace(".jpg", ".txt"), "a") as img_txt:
            img_txt.write(index + " " + " " + str(obj_mid_x_rel) + " " + str(obj_mid_y_rel) + " " +  str(obj_width_rel) + " " + str(obj_height_rel) + "\n")
        if index == "1" and "dark_" + filename.replace(".jpg", ".txt") not in os.listdir(path + "/test"):
            with open("test.txt", "a") as train_txt:
                train_txt.write("data/obj/test/dark_" + filename + "\n")
                train_txt.write("data/obj/test/bright_" + filename + "\n")
            shutil.copy(path + "/test/" + filename.replace(".jpg", ".txt"), path + "/test/dark_" + filename.replace(".jpg", ".txt"))
            shutil.copy(path + "/test/" + filename.replace(".jpg", ".txt"), path + "/test/bright_" + filename.replace(".jpg", ".txt"))
    elif filename in os.listdir(path + "/train"):
        with open(path + "/train/" + filename.replace(".jpg", ".txt"), "a") as img_txt:
            img_txt.write(index + " " + " " + str(obj_mid_x_rel) + " " + str(obj_mid_y_rel) + " " +  str(obj_width_rel) + " " + str(obj_height_rel) + "\n")
        shutil.copy(path + "/train/" + filename.replace(".jpg", ".txt"), path + "/test/" + filename.replace(".jpg", ".txt"))
        if index == "1" and "dark_" + filename.replace(".jpg", ".txt") not in os.listdir(path + "/test"):
            shutil.copy(path + "/train/" + filename.replace(".jpg", ".txt"), path + "/test/dark_" + filename.replace(".jpg", ".txt"))
            shutil.copy(path + "/train/" + filename.replace(".jpg", ".txt"), path + "/test/bright_" + filename.replace(".jpg", ".txt"))    
            with open("test.txt", "a") as test_txt:
                test_txt.write("data/obj/test/dark_" + filename + "\n")
                test_txt.write("data/obj/test/bright_" + filename + "\n")        
        with open("test.txt", "a") as test_txt:
            test_txt.write("data/obj/test/" + filename + "\n")    
    else:
        if index == "1":      
            with open(path + "/test/dark_" + filename.replace(".jpg", ".txt"), "w+") as img_txt:
                img_txt.write(index + " " + " " + str(obj_mid_x_rel) + " " + str(obj_mid_y_rel) + " " +  str(obj_width_rel) + " " + str(obj_height_rel) + "\n")
            with open(path + "/test/bright_" + filename.replace(".jpg", ".txt"), "w+") as img_txt:
                img_txt.write(index + " " + " " + str(obj_mid_x_rel) + " " + str(obj_mid_y_rel) + " " +  str(obj_width_rel) + " " + str(obj_height_rel) + "\n")
            with open("test.txt", "a") as test_txt:
                test_txt.write("data/obj/test/dark_" + filename + "\n")
                test_txt.write("data/obj/test/bright_" + filename + "\n")
        with open(path + "/test/" + filename.replace(".jpg", ".txt"), "w+") as img_txt:
            img_txt.write(index + " " + " " + str(obj_mid_x_rel) + " " + str(obj_mid_y_rel) + " " +  str(obj_width_rel) + " " + str(obj_height_rel) + "\n")
        with open("test.txt", "a") as test_txt:
            test_txt.write("data/obj/test/" + filename + "\n")
    shutil.copy(row["path"], newpath)