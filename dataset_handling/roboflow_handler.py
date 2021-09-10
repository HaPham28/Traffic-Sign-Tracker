# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 15:55:40 2021

@author: Ha V. Pham
"""

import os

for filename in os.listdir():
    if filename.endswith(".txt"):
        with open(filename, 'r+') as f:
            f.seek(0,0)
            content = f.read().splitlines()
            print(content)
            for i in range(len(content)):
                if content[i][0] == '0':  content[i] = '2' + content[i][1:] + '\n'
                elif content[i][0] == '4': content[i] = '0' + content[i][1:] + '\n'
                elif content[i][0] == '5': content[i] = '1' + content[i][1:] + '\n'
                else: content[i] = '3' + content[i][1:] + '\n'
            print("****", content)
            f.seek(0,0)
            f.writelines(content)
            f.seek(0,0)
            content = f.read().splitlines()
            print(content)
            with open("train.txt", "a") as train_txt:
                train_txt.write("data/obj/train/" + filename.replace(".jpg", ".txt") + "\n")
            
            