import fitz
import subprocess
import time
import json
import pandas as pd
from copy import deepcopy
import re
import csv
import sys
from io import StringIO

import cv2 as cv
import numpy as np

from PIL import Image
from pprint import pprint

import pytesseract

import easyocr

reader = easyocr.Reader(['ru'])
reader_num = easyocr.Reader(['ru','en'])

reader.lang_char += "«»"



with open("config_tmp.json", "r") as json_file:
    tmp1conf = json.load(json_file)

# Tesseract invocation command (Windows version)
# Assume: language English. Detect more languages by add e.g. '+deu' for German.
# Assume: text represents one line (--psm 7)
# Note: Language mix spec increases duration by >40% - only use when needed!
tess = "tesseract stdin stdout --psm 7 -l rus"



def get_tessocr(page, bbox):
    """Return OCR-ed span text using Tesseract.
    Args:
        page: fitz.Page
        bbox: fitz.Rect or its tuple
    Returns:
        The OCR-ed text of the bbox.
    """
    global ocr_time, pix_time, tess, mat
    # Step 1: Make a high-resolution image of the bbox.
    t0 = time.perf_counter()
    pix = page.get_pixmap(
        colorspace=fitz.csGRAY,  # we need no color
        matrix=mat,
        clip=bbox,
    )
    image = pix.tobytes("png")  # make a PNG image
    t1 = time.perf_counter()
    # Step 2: Invoke Tesseract to OCR the image. Text is stored in stdout.
    rc = subprocess.run(
        tess,  # the command
        input=image,  # the pixmap image
        stdout=subprocess.PIPE,  # find the text here
        shell=True,
    )

    # because we told Tesseract to interpret the image as one line, we now need
    # to strip off the line break characters from the tail.
    text = rc.stdout.decode()  # convert to string
    text = text[:-3]  # remove line end characters
    t2 = time.perf_counter()
    ocr_time += t2 - t1
    pix_time += t1 - t0
    return text

def matching():
    pass

def milestonefinde(fitz_data, param):
    milse = ""
    const_name = -1
    k = 0
    for line in fitz_data:
        match = re.search(param, line.lower())
        if match:
            if const_name == -1:
                const_name = k
            milse = line + "\n" + milse
        k+=1
    return milse, const_name

def model(img_path, whitelist):
    return pytesseract.image_to_string(img_path,
                                config="-c tessedit_char_blacklist=йэ\/"
                                # "user_words_suffix=user-words user_patterns_suffix=user-patterns"
                                       " --psm 3 --oem 2 "
                                       "--user-patterns /usr/local/Cellar/tesseract/5.2.0/share/tessdata/rus.user-patterns "
                                       "--user-words /usr/local/Cellar/tesseract/5.2.0/share/tessdata/rus.user-words "
                                       "-l rus".format(whitelist))


def titul_recgnize(data, param, fitz_data):

    tess = re.sub(r"\n\b", " ", pytesseract.image_to_string(data , config="--psm 3 -l rus+eng"))
    pprint(tess)
    print("\nfitzdata")
    for b in fitz_data:
        for l in b["lines"]:
            for line in l["spans"]:
                print(line["text"])

    result = {
        "documentCipher": {
            "fitz": "",
            "tess": ""
        },
        "documentName": {
            "fitz": "",
            "tess": ""
        },
        "constructionName": {
            "fitz": "",
            "tess": ""
        },
        "changeNumber": {
            "fitz": "",
            "tess": ""
        },
        "milestone": {
            "fitz": "",
            "tess": ""
        },
        "inventoryNumber": {
            "fitz": "",
            "tess": ""
        },
        "documentDate": {
            "fitz": "",
            "tess": ""
        }
    }

    tmp = None

    if fitz_data:
        fitz_lines = []
        for b in fitz_data:
            for l in b["lines"]:
                for line in l["spans"]:
                    fitz_lines.append(line["text"].strip())
        inv = fitz_lines[-1]
        match = re.search(param["inventoryNumber"]["reg"], inv)
        if match:
            tmp = 2
            result["inventoryNumber"]["fitz"] = match[0]
        if tmp is None:
            for l in fitz_lines:
                if "проектная документация" == l.lower().strip():
                    tmp = 2
                    break

        documentCipher_ind = None
        for line in range(len(fitz_lines)):
            match = re.search(param["documentCipher"]["reg"], fitz_lines[line])
            if match:
                documentCipher_ind = line-1
                match = re.search(param["changeNumber"]["reg"], fitz_lines[line].lower())
                if match:
                    result["changeNumber"]["fitz"] = match[1]
                result["documentCipher"]["fitz"] = fitz_lines[line]
                if tmp==2:
                    match = re.search(r"20\d{2}", fitz_lines[1])
                    if match:
                        result["documentDate"]["fitz"] += fitz_lines[1]
                for i in range(1,4):
                    match = re.search(param["changeNumber"]["reg"], fitz_lines[line+i].lower())
                    if match:
                        result["changeNumber"]["fitz"] = match[1]
                    if tmp == 2 and "том" in fitz_lines[line+i].lower():
                        result["documentName"]["fitz"] += fitz_lines[line+i]
                        break
                break
        constructionName_ind = None
        if documentCipher_ind:
            milestone = False
            docnamelen = 0
            print("wich tmp?")
            print(tmp)
            construction = False
            for line in reversed(range(documentCipher_ind)):
                # if str(fitz_lines[line]).replace(' ', '') == "\n":
                #     pass
                # else:
                if tmp == 2:
                    print(fitz_lines[line])
                    match = re.search(param["milestone"]["reg"], fitz_lines[line].lower())
                    if match:
                        result["milestone"]["fitz"] = fitz_lines[line] + "\n" + result["milestone"]["fitz"]
                        milestone = True
                    elif milestone and str(fitz_lines[line]).replace(' ', '') == "\n":
                        pass
                    elif construction:
                        milestone = True
                        construction = False
                    elif milestone:
                        space = 0
                        constructionLen = 0
                        k = 0
                        while True:
                            if str(fitz_lines[line-k]).replace(' ', '') == "\n":
                                space +=1
                            if len(fitz_lines[line-k]) > 3:
                                result["constructionName"]["fitz"] = fitz_lines[line-k] + " " + result["constructionName"]["fitz"]
                                constructionLen+=1
                            # elif len(fitz_lines[line-1]) > 3:
                            #     result["constructionName"]["fitz"] = fitz_lines[line-1]
                            if space >= 3 or constructionLen >= 4 or k >= 8:
                                break
                            k+=1
                        break
                    else:
                        if docnamelen<=10 and str(fitz_lines[line+1]).replace(' ', '').replace('\n', '') != "":
                            result["documentName"]["fitz"] = fitz_lines[line+1] + " " + result["documentName"]["fitz"]
                            docnamelen += 1
                    construction = True if "проектная документация" == fitz_lines[line].lower().strip() else False
                elif tmp == 1:
                    match = re.search("«.+»", fitz_lines[line])
                    if match:
                        result["constructionName"]["fitz"] = fitz_lines[line]
                        break
                else:
                    match = re.search(param["milestone"]["reg"], fitz_lines[line].lower())
                    if match:
                        result["milestone"]["fitz"] = fitz_lines[line] + "\n" + result["milestone"]["fitz"]
                        milestone = True
                    elif milestone:
                        result["constructionName"]["fitz"] = fitz_lines[line]
                        break
                    else:
                        if docnamelen<=6:
                            result["documentName"]["fitz"] = fitz_lines[line+1] + " " + result["documentName"]["fitz"]
                            docnamelen += 1
                    match = re.search("«.+»", fitz_lines[line])
                    if match:
                        result["constructionName"]["fitz"] = fitz_lines[line]
                        break
            if result["milestone"]["fitz"] == "":
                result["milestone"]["fitz"],  construction_ind = milestonefinde(fitz_lines, param["milestone"]["reg"])


    documentCipher_ind = None
    tess_ilnes = tess.split("\n")
    if tmp is None:
        for l in tess_ilnes:
            if "проектная документация" in l.lower().strip():
                tmp = 2
                break
    for line in range(len(tess_ilnes)):
        match = re.search(param["documentCipher"]["reg"], tess_ilnes[line])
        if match:
            documentCipher_ind = line-1
            match = re.search(param["changeNumber"]["reg"], tess_ilnes[line].lower())
            if match:
                result["changeNumber"]["tess"] = match[1]
            result["documentCipher"]["tess"] = tess_ilnes[line]
            if tmp==2:
                match = re.search(r"20\d{2}", tess_ilnes[1])
                if match:
                    result["documentDate"]["tess"] += tess_ilnes[1]
            for i in range(1,4):
                if tmp == 2 and "том" in tess_ilnes[line+i].lower():
                    result["documentName"]["tess"] += tess_ilnes[line+i]
                    break
            break
    constructionName_ind = None
    if documentCipher_ind:
        milestone = False
        docnamelen = 0
        construction = False
        for line in reversed(range(documentCipher_ind)):

            if tmp == 2:
                # print(tess_ilnes[line])
                match = re.search(param["milestone"]["reg"], tess_ilnes[line].lower())
                if match:
                    print("milestone add")
                    result["milestone"]["tess"] = tess_ilnes[line] + "\n" + result["milestone"]["tess"]
                    milestone = True
                elif construction:
                    print("construction")
                    milestone = True
                    construction = False
                elif milestone:
                    print("milestone")
                    space = 0
                    constructionLen = 0
                    k = 0
                    while True:
                        if str(tess_ilnes[line-k]).replace(' ', '') == "\n":
                            space += 1
                        if space >= 2 or constructionLen >= 4 or k >= 8:
                            break
                        if len(tess_ilnes[line-k]) > 3:
                            result["constructionName"]["tess"] = tess_ilnes[line-k] + " " + result["constructionName"]["tess"]
                            constructionLen+=1
                        # elif len(fitz_lines[line-1]) > 3:
                        #     result["constructionName"]["fitz"] = fitz_lines[line-1]

                        k+=1
                    break
                # match = re.search(param["milestone"]["reg"], tess_ilnes[line].lower())
                # if match:
                #     result["milestone"]["tess"] = tess_ilnes[line] + "\n" + result["milestone"]["tess"]
                #     milestone = True
                # elif milestone:
                #     if len(tess_ilnes[line]) > 3:
                #         result["constructionName"]["tess"] = tess_ilnes[line]
                #     elif len(tess_ilnes[line-1]) > 3:
                #         result["constructionName"]["tess"] = tess_ilnes[line-1]
                #     break
                else:
                    if docnamelen<=6:
                        result["documentName"]["tess"] = tess_ilnes[line+1] + " " + result["documentName"]["tess"]
                        docnamelen += 1

                construction = True if "проектная документация" == tess_ilnes[line].lower().strip() else False
            elif tmp == 1:
                match = re.search("«.+»", tess_ilnes[line])
                if match:
                    result["constructionName"]["tess"] = tess_ilnes[line]
                    break
            else:
                match = re.search(param["milestone"]["reg"], tess_ilnes[line].lower())
                if match:
                    result["milestone"]["tess"] = tess_ilnes[line] + "\n" + result["milestone"]["tess"]
                    milestone = True
                elif milestone:
                    result["constructionName"]["tess"] = tess_ilnes[line]
                    break
                else:
                    if docnamelen<=6:
                        result["documentName"]["tess"] = tess_ilnes[line+1] + " " + result["documentName"]["tess"]
                        docnamelen += 1
                match = re.search("«.+»", tess_ilnes[line])
                if match:
                    result["constructionName"]["tess"] = tess_ilnes[line]
                    break

    return result


def stamp_recognize(data, param, entity, w, h, alt_par = None, fitz_data=None):
    print("start stamp")
    result = {entity: {"tess": "",
                       "easy": "",
                       "fitz": ""}}

    if param["config"]:
        tess = pytesseract.image_to_data(data,
                                         config=param["config"])
    else:
        tess = pytesseract.image_to_data(data, lang=param["lang"])
    with open("part.tsv", "w") as file:
        file.write(tess)
    tess = pd.read_table("part.tsv")
    tess = tess[tess["conf"] >= 40]
    easy = reader.readtext(data)
    if param["reg"]:
        for te in tess["text"].values:
            te = str(te)
            match = re.search(param["reg"], te)
            if match:
                if result[entity]["tess"] == "":
                    result[entity]["tess"] = match[0]
        for ea in easy:
            if ea[2] >= .5:
                match = re.search(param["reg"], ea[1])
                if match:
                    if result[entity]["easy"] == "":
                        result[entity]["easy"] = match[0]
        if fitz_data:
            if alt_par:
                tmpar = alt_par
            else:
                tmpar = param["rect"]
            for b in fitz_data:
                for l in b["lines"]:
                    for line in l["spans"]:
                        # print(line)
                        if (line["bbox"][0]-w)*6 >= tmpar[0][0] and (line["bbox"][1]-h)*6 >= tmpar[0][1] \
                                and (line["bbox"][2]-w)*6 <= tmpar[1][0] and (line["bbox"][3]-h)*6 <= tmpar[1][1]:
                            match = re.search(param["reg"], line["text"])
                            if match:
                                if result[entity]["fitz"] == "":
                                    result[entity]["fitz"] = match[0]

            if entity=="stage" and result[entity]["fitz"] == "":
                x,y = 0, 0
                for b in fitz_data:
                    for l in b["lines"]:
                        for line in l["spans"]:
                            if line["text"].lower().strip() == "стадия":
                                x, y = line["bbox"][0], line["bbox"][3]
                if x > 0 and y > 0:
                    for b in fitz_data:
                        for l in b["lines"]:
                            for line in l["spans"]:
                                if line["bbox"][0] >= x and  line["bbox"][1]>=y:
                                    match = re.search(param["reg"], line["text"])
                                    if match:
                                        if result[entity]["fitz"] == "":
                                            result[entity]["fitz"] = match[0]
                                        break



    else:
        text = re.sub("[^\w+ .-«»]", " ", " ".join(tess["text"]))
        text = re.sub(' +', ' ', text)
        if len(text) > 3:
            result[entity]["tess"] = text.strip()

        text = ""
        for ea in easy:
            text += " " + ea[1]

        text = re.sub("[^\w+ .-«»]", " ", text)
        text = re.sub(' +', ' ', text)
        if len(text) > 3:
            result[entity]["easy"] = text.strip()

        if fitz_data:
            if alt_par:
                tmpar = alt_par
            else:
                tmpar = param["rect"]
            text = ""
            for b in fitz_data:
                for l in b["lines"]:
                    for line in l["spans"]:
                        # print(line)
                        if (line["bbox"][0]-w)*6 >= tmpar[0][0] and (line["bbox"][1]-h)*6 >= tmpar[0][1] \
                                and (line["bbox"][2]-w)*6 <= tmpar[1][0] and (line["bbox"][3]-h)*6 <= tmpar[1][1]:
                            text+=" "+line["text"]
            text = re.sub("[^\w+ .-«»]", " ", text)
            text = re.sub(' +', ' ', text)
            if len(text) > 3:
                result[entity]["fitz"] = text.strip()

    return result[entity]

def block_search(blocks, page):
    ocr_time = 0
    pix_time = 0
    bb = []
    for b in blocks:
        for l in b["lines"]:
            for s in l["spans"]:
                # print(s)
                text = s["text"]
                # print(text)
                if chr(65533) in text:  # invalid characters encountered!
                    # invoke OCR
                    print("before: '%s'" % text)
                    text1 = text.lstrip()
                    sb = " " * (len(text) - len(text1))  # leading spaces
                    text1 = text.rstrip()
                    sa = " " * (len(text) - len(text1))  # trailing spaces
                    new_text = sb + get_tessocr(page, s["bbox"]) + sa
                    s["text"] = new_text
                    print(" after: '%s'" % new_text)
                bb.append(s)
    return bb