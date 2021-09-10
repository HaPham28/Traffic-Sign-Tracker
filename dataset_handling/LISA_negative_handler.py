# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 19:09:39 2021

@author: Ha V. Pham
"""
import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

li = list()
for filename in os.listdir():
    if filename.endswith(".png"):
        if int(float(filename[6:11]))%2 == 0:
           li.append(filename)

df = pd.DataFrame(li, columns = ['name'])
train, test = train_test_split(df, test_size=0.1)

trainpath = "train"
if not os.path.exists(trainpath): #create a new folder for new images if not exists
    os.makedirs(trainpath)
for index, row in train.iterrows():
    filename = row["name"]
    newpath = trainpath + "/"+ filename.replace(".png", ".jpg")
    shutil.copy(filename, newpath)
    with open("train/" + filename.replace(".png", ".txt"), "w+") as img_txt:
        img_txt.write("")
    with open("train.txt", "a") as train_txt:
        train_txt.write("data/obj/train/" + filename + "\n")    
        
testpath = "test"
if not os.path.exists(testpath): #create a new folder for new images if not exists
    os.makedirs(testpath)
for index, row in test.iterrows():
    filename = row["name"]
    newpath = testpath + "/"+ filename.replace(".png", ".jpg")
    shutil.copy(filename, newpath)
    with open("test/" + filename.replace(".png", ".txt"), "w+") as img_txt:
        img_txt.write("")
    with open("test.txt", "a") as test_txt:
        test_txt.write("data/obj/test/" + filename + "\n")