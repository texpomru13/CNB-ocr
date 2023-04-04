import sys
import pickle
import os
import pytesseract
import re
import dataobjects

import fitz
import json
from pprint import pprint

from stamp import stamp_recognize, titul_recgnize, reader, reader_num
from line import linecrop
from dateinst import fineins, findedate

import cv2 as cv
import numpy as np

import cv2

import re
import json
import preprocessing
from PIL import Image
from PIL import ImageEnhance

from autocorrect import Speller
spell = Speller('ru', only_replacements=True)

import sys
if sys.version_info[0] < 3:
    from StringIO import StringIO
else:
    from io import StringIO


from transformers import AutoModelForSeq2SeqLM, T5TokenizerFast

# Зададим название выбронной модели из хаба
MODEL_NAME = 'path/to/model'
MAX_INPUT = 256

# Загрузка модели и токенизатора
tokenizer = T5TokenizerFast.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

with open("config_tmp.json", "r") as json_file:
    tmp1conf = json.load(json_file)
    param = tmp1conf


def get_cropped_image(image, x, y, w, h):
    cropped_image = image[y:h, x:w]
    return cropped_image

def preproc(img_path):

    image = cv2.imread(img_path).astype("uint8")

    thresh = image
    thresh = preprocessing.removeline(image)
    # thresh = preprocessing.get_grayscale(thresh)
    try:
        cv2.imwrite(img_path, thresh)
    except:
        return 1

    #
    # # Opens the image file
    image = Image.open(img_path)
    #
    # # shows image in image viewer
    # # Enhance Sharpness
    curr_sharp = ImageEnhance.Sharpness(image)
    new_sharp = 4
    #
    # # Sharpness enhanced by a factor of 8.3
    thresh = curr_sharp.enhance(new_sharp)
    thresh.save(img_path)

def extrstamp_onlytess(path, startpage= 0):
    doc = fitz.open(path)
    mat = fitz.Matrix(2, 2)
    filename = ".".join(doc.name.split("/")[-1].split(".")[:-1])
    mpage = 7
    img_path = None
    k = 0
    for page in doc:
        rect = page.rect  # the page rectangle
        mp = (rect.tl + rect.br) / 2  # its middle point, becomes top-left of clip
        fitz.Point(0, rect.height / 3)
        find = {"d":0,"i":0,"l":0,"s":0,"lv":0,"k":0,"c":0}
        check = {"d":"(^| )(д|а)(а|е)(т|ш)а", "i":"(`|^| |‚)(и|й)(з|в)м.? ?", "l":"(^| )лист", "s":"(^| )с(т|п)(а|о)(д|в|б)ия",
                 "lv":"(^| )(л|и)?истов", "k":"(^| )к?кол( |.)?у?", "c": param["titular"]["documentCipher"]["reg"]}
        w = rect.width - 590
        h = rect.height - 195
        mp1 = fitz.Point(w, h)
        mp2 = fitz.Point(rect.width, rect.height)
        clip = fitz.Rect(mp1, mp2)
        pix = page.get_pixmap(matrix=mat, clip=clip)
        img_path = "tmpstamp.png"
        pix.save(img_path)
        if preproc(img_path) == 1:
            os.remove(img_path)
            continue
        whitelist = "листдазмявкоу.№|"
        whitelist += whitelist.upper()
        text = re.sub(r" ", " ", pytesseract.image_to_string(img_path,
                                                             config="-c tessedit_char_blacklist=йэ\/"
                                                             # "user_words_suffix=user-words user_patterns_suffix=user-patterns"
                                                                    " --psm 3 --oem 2 "
                                                                    "--user-patterns /usr/local/Cellar/tesseract/5.2.0/share/tessdata/rus.user-patterns "
                                                                    "--user-words /usr/local/Cellar/tesseract/5.2.0/share/tessdata/rus.user-words "
                                                                    "-l rus".format(whitelist))) #\n\b -c tessedit_char_whitelist={}

        text= re.sub("['\\|\\[{}_]", " ", text)
        text= re.sub(" +", " ", text)
        text= re.sub("-+", "-", text)

        text = text.lower()
        print(text)
        lines = text.split("\n")
        ch = 0
        for line in lines:
            for che in check:
                if re.search(check[che], line):
                    if find[che] <1:
                        if che == "i" and "внес" in line:
                            continue
                        # print(che)
                        if che in "ik":
                            find[che] +=2
                        else:
                            find[che] +=1
        ch= sum(find.values())
        find = {"d":0,"i":0,"l":0,"s":0,"lv":0,"k":0,"c":0}

        print(k)
        print("tess ch - "+ str(ch))
        if ch >= 5:
            # print(text)
            return img_path, k
        else:
            os.remove(img_path)
        # page.set_rotation(90)
        # w = rect.width - 880
        # h = 0
        # mp1 = fitz.Point(w, h)
        # mp2 = fitz.Point(rect.width, rect.height-720)
        # clip = fitz.Rect(mp1, mp2)
        # pix = page.get_pixmap(matrix=mat, clip=clip)
        # img_path = "tmpstamp.rot.png"
        # try:
        #     pix.save(img_path)
        # except:
        #     continue
        #
        # if preproc(img_path) == 1:
        #     os.remove(img_path)
        #     continue
        # whitelist = "листдазмявкоу.№|"
        # whitelist += whitelist.upper()
        # text = re.sub(r" ", " ", pytesseract.image_to_string(img_path,
        #                                                      config="-c tessedit_char_blacklist=йэ\/"
        #                                                      # "user_words_suffix=user-words user_patterns_suffix=user-patterns"
        #                                                             " --psm 3 --oem 2 "
        #                                                             "--user-patterns /usr/local/Cellar/tesseract/5.2.0/share/tessdata/rus.user-patterns "
        #                                                             "--user-words /usr/local/Cellar/tesseract/5.2.0/share/tessdata/rus.user-words "
        #                                                             "-l rus".format(whitelist))) #\n\b -c tessedit_char_whitelist={}
        #
        # text= re.sub("['\\|\\[{}_]", " ", text)
        # text= re.sub(" +", " ", text)
        # text= re.sub("-+", "-", text)
        #
        # text = text.lower()
        # print(text)
        # lines = text.split("\n")
        # ch = 0
        # for line in lines:
        #     for che in check:
        #         if re.search(check[che], line):
        #             if find[che] <1:
        #                 if che == "i" and "внес" in line:
        #                     continue
        #                 # print(che)
        #                 if che in "ik":
        #                     find[che] +=2
        #                 else:
        #                     find[che] +=1
        # ch= sum(find.values())
        # find = {"d":0,"i":0,"l":0,"s":0,"lv":0,"k":0,"c":0}
        #
        # print(k)
        # print("tess ch - "+ str(ch))
        # if ch >= 5:
        #     # print(text)
        #     return img_path, k
        # else:
        #     os.remove(img_path)
        if  k >= mpage:
            return img_path, None
            # break
        k+=1
        # if ch >= 5:
        #     # print(text)
        #     break
    return img_path, None

def findeStamp(file):
    img_path, list_num = extrstamp_onlytess(file)
    try:
        image = Image.open(img_path)
    except:
        return {}
    width, height = image.size
    newsize = (width*5, height*5)
    im1 = image.resize(newsize)
    if type(list_num) == int:
        fold = file.split("/")[-2]
    else:
        fold = "none_"+file.split("/")[-2]
    os.makedirs("stimg/"+fold, exist_ok=True)
    im1.save("stimg/"+fold+"/"+file.split("/")[-1].replace(".pdf", ".png"))

param_stamp ={"d": "((^| )(й|ч|д|л|йп)(а|о|е)(т|ш|г)(а|о))|(ата)", "i": "(`|^| |‚)(и|й)(з|в)?м(,|.)? ?",
              "s":"((^| )(с|\\()?(т|п)(а|о|9)(б|д|й)?(д|в|б)?(у|ы|и)я)|(внутри)|(стай)", "c": param["titular"]["documentCipher"]["reg"]}
check = "((^| )?(й|ч|д|л|йп)(а|е)(т|ш)а)|((`|^| |‚)(и|й)(з|в)м.?)|((^| )?к?кол( |.)?у?)"
black = "{};*!®@[]"
def stampDataEextract(img):
    data = pytesseract.image_to_data(img,
                                     config="-c tessedit_char_blacklist={} --psm 3 "
                                            "-l rus".format(black), output_type='data.frame')


    data["right"] = data["left"]+data["width"]
    data["down"] = data["top"]+data["height"]
    data = data.dropna(0)
    data = data[["text","left","right","top","down","width","height","block_num","par_num"]]
    data.to_csv("tmp.csv", index=False)
    return data

def stamp_rasb(data, rot=False):
    place = {}
    ret = {}
    for par in param_stamp:
        if par == "c":
            block = data["block_num"].drop_duplicates().values
            var = []
            for b in block:
                block_data = data[data["block_num"] == b]
                left = 2000
                if "d" in place:
                    left = place["d"][0] + place["d"][2]
                if block_data["left"].min() > left:
                    tmp_text = block_data["text"].sum().replace(",", ".")
                    match = re.search(param_stamp[par], tmp_text)
                    if re.search(check, tmp_text.lower()):
                        continue
                    if match:
                        var = [[block_data["left"].min(), block_data["top"].min(), block_data["width"].sum(),
                                block_data["height"].max(), block_data["text"].sum().replace(",", ".")]]
                        break

            if var == []:
                var = data[data["text"].map(str.lower).str.contains(param_stamp[par], regex=True)][["left", "top", "width", "height", "text", ]].values
            else:
                place[par] = var[0]
                continue

            if len(var)>0:
                var = var[0]
                var = data[(data["left"]>=var[0]) & (data["top"]>=var[1]-5) &
                           (data["top"]+data["height"]<=var[1]+var[3]+70)][["left", "top", "width", "height", "text"]]
                place[par] = [var["left"].min(), var["top"].min(), var["width"].sum(), var["height"].max(), var["text"].sum()]
        elif par == "i":
            var = data[(data["text"].map(str.lower).str.contains(param_stamp[par], regex=True)) &
                       (data["left"]<=1000)][["left", "top", "width", "height", "text"]].values
            # print(var)
            if len(var)>0:
                place[par] =var[0]
        elif par == "s":
            var = data[(data["text"].map(str.lower).str.contains(param_stamp[par], regex=True)) &
                       (data["left"]>2000)][["left", "top", "width", "height", "text"]].values
            if len(var)>0:
                coord =var[0]
                place[par] = coord
            else:
                var = data[(data["text"].map(str.lower).str.contains("^с$", regex=True)) &
                           (data["left"]>4000)][["left", "top", "width", "height", "text"]].values
                if len(var)>0:
                    coord =var[0]
                    coord[2] += 140
                    place[par] = coord
        elif par == "d":
            var = data[(data["text"].map(str.lower).str.contains(param_stamp[par], regex=True)) &
                       (data["left"]>1000) &
                       (data["right"]<3000)][["left", "top", "width", "height", "text"]].values
            # print(var)
            if len(var)>0:
                place[par] =var[0]
        else:
            var = data[(data["text"].map(str.lower).str.contains(param_stamp[par], regex=True)) &
                       (data["left"]>200)][["left", "top", "width", "height", "text"]].values
            # print(var)
            if len(var)>0:
                place[par] =var[0]
    construction = [0,0,0,0,""]
    doc = [0,0,0,0,""]
    inst = [0,0,0,0,""]
    print(place)
    if "c" in place:
        ret["c"] = place["c"]

        if rot:
            inst[0] = place["s"][0]+place["c"][2]
            inst[2] = place["c"][2]
            inst[1] = place["c"][1]-20
            inst[3] = place["c"][3]+50
            doc[0] = place["s"][0]-3
            doc[2] = int(place["c"][2]*1.6)
            doc[3] = place["c"][3]*4
        else:
            construction[1] = place["c"][1]+place["c"][3]
            construction[0] = place["c"][0]-place["c"][2]/2
            construction[2] = place["c"][2]*2.3
            construction[3] = int(place["c"][3]*1.6)
            inst[2] = place["c"][2]*2
            inst[3] = int(place["c"][3]*8)
    if "i" in place:
        if rot:
            pass
        else:
            doc[1] = place["i"][1]+place["i"][3]+10
        ret["i"] = data[(data["left"] >= place["i"][0]) & (data["top"] <= place["i"][1])
                        & (data["left"]+data["width"] <= place["i"][0]+place["i"][2])
                        & (data["text"].str.contains("^\\d$", regex=True))][["left", "top", "width", "height", "text"]].values
    if "d" in place:
        if rot:
            doc[1] = place["d"][1]+place["d"][3]+5
            doc[0] = place["d"][0]+place["d"][2]+20
            doc[3] = place["d"][3]*6
        else:
            construction[0] = place["d"][0]+place["d"][2]+20
            doc[1] = place["d"][1]+place["d"][3]+10
            doc[0] = place["d"][0]+place["d"][2]+20
            doc[3] = place["d"][3]*6
        ret["d"] = data[(data["left"]+100 >= place["d"][0]) & (data["left"]+data["width"]-100 <= place["d"][0]+place["d"][2])
                        & (data["text"].str.contains("(1[0-2]|0?[1-9](.|,|:|;))?(3[01]|[12][0-9]|0?[1-9])(.|,|:|;)(0?[1-35]|((3[01]|[12][0-9]|0?[1-9])|0[1-9])/[0-9]{4})",
                                                     regex=True))][["left", "top", "width", "height", "text"]].values
        print("d")
        pprint(ret["d"])
        pprint(place["d"])
    if "s" in place:

        if rot:
            pass
        else:
            construction[3] = place["s"][1]-5 - construction[1]
            # construction[2] = place["s"][0]-5 - construction[0]
            doc[1] = place["s"][1]-5
            inst[1] = place["s"][1]+place["s"][3]+200
            inst[0] = place["s"][0]
            doc[2] = place["s"][0] - doc[0] - 10
            doc[3] = place["s"][3]*8
        ret["s"] = data[(data["left"] >= place["s"][0]) & (data["top"] > place["s"][1]+place["s"][3]) & (data["top"] <= place["s"][1]+250)
                        & (data["left"]+data["width"] <= place["s"][0]+place["s"][2])
                        & (data["text"].str.contains("^[PpРрПплЛ]$", regex=True))][["left", "top", "width", "height", "text"]].values

    if construction[0] != 0:
        coord = []
        for i in construction[:-1]:
            if i == 0:
                coord.append(9.5)
            else:
                coord.append(i)
        text = " ".join(data[(data["left"] >= coord[0]) & (data["top"] >= coord[1]-5) & (data["top"]+data["height"]<=coord[1]+coord[3]+70)
                             & (data["left"]+data["width"] <= coord[0]+coord[2])]["text"].values)
        if 9.5 not in coord:
            coord[2] = data["right"].max() - coord[0]
        ret["con"] = [coord,text]

    if doc[0] != 0:
        coord = []
        for i in doc[:-1]:
            if i == 0:
                coord.append(400)
            else:
                coord.append(i)
        text = " ".join(data[(data["left"] >= coord[0]) & (data["top"] > coord[1]) & (data["top"]+data["height"]<=coord[1]+coord[3]+70)
                             & (data["left"]+data["width"] <= coord[0]+coord[2])]["text"].values)
        ret["doc"] = [coord,text]

    if inst[0] != 0:
        coord = []
        for i in inst[:-1]:
            if i == 0:
                coord.append(400)
            else:
                coord.append(i)
        text = " ".join(data[(data["left"] >= coord[0]) & (data["top"] > coord[1]) & (data["top"]+data["height"]<=coord[1]+coord[3]+70)
                             & (data["left"]+data["width"] <= coord[0]+coord[2])]["text"].values)
        ret["inst"] = coord+[text]
        ret["inst"][2] = data["right"].max() - ret["inst"][0]
        ret["inst"][3] = data["down"].max() - ret["inst"][1]
        ret["inst"][0] -= 20
    fp = {}
    for i in ret:
        text = ""
        if len(ret[i]) == 2:
            text = ret[i][1]
        elif len(ret[i]) == 5:
            text = fp[i] = ret[i][4]
        elif len(ret[i]) != 0:
            text = fp[i] = ret[i][0][4]
        try:
            text = re.sub("([\\|%#\\^=]|(Ф|ф)ормат (А|а)4|(Л|л)ист)","", text)
        except:
            pass
        fp[i] = text
    # pprint(fp)
    return ret, place

def cropData(image, ret, place):
    allparam = {}

    # image = Image.open(img_path)
    for i in ret:
        text = ""
        crop = True
        pl = False
        # print(ret[i])
        if i == "d" and ret[i] != [] and len(ret[i]) != 0:
            left, top, right, bottom = ret[i][0][:4]
        elif i == "i" and ret[i] != [] and len(ret[i]) != 0:
            left, top, right, bottom = ret[i][0][:4]
        elif len(ret[i]) == 2:
            # print(ret[i])
            left, top, right, bottom = ret[i][0]
        elif len(ret[i]) == 5:
            left, top, right, bottom = ret[i][:4]
        elif len(ret[i]) != 0:
            left, top, right, bottom = ret[i][0][:4]
        else:
            if i == "i":
                left, top, right, bottom = place[i][:4]
            elif i == "d":
                left, top, right, bottom = place[i][:4]
            elif i == "s":
                left, top, right, bottom = place[i][:4]
            else:
                crop = False
            if crop:
                pl = True

        if crop:
            right += left
            heigt = bottom
            bottom += top
            if i == "c":
                left -= 50
                right += 150
                top -= 20
                bottom += 20
            if i == "d" and pl:
                bottom += heigt*2
                top += heigt
                top += 10
                left -= 35
                right += 35
            elif i == "d":
                top -= 20
                bottom += 20
                left -= 35
                right += 35
            if i == "inst":
                left -= 35
                right += 35
            if i == "doc":
                bottom += 100
            if (i == "i" ) and pl: #or i == "d"
                bottom -= heigt
                bottom -= 20
                top -= heigt
                top -= 80
            if i == "s" and pl:
                bottom += heigt+180
                top += heigt
                top += 10


            # print(i)
            # print((left, top, right, bottom))
            try:
                im1 = image.crop((left, top, right, bottom))
                im1.save("shab/"+i+".png")
                allparam[i] = im1
                # if i == "d":
                #     im1 = image.crop((left, top, right, bottom))
            except:
                pass
    return allparam

def extractCrop(allpram):
    stamp_res = {}
    for t in allpram:
        black = ""
        white = ""
        psm = "6"
        if t == "i":
            white = "123456789"
            psm = "10"
        elif t == "d":
            white = "1234567890."
            psm = "3"
        elif t == "s":
            white = "РрПпPp"
            psm = "10"
        elif t == "c":
            black = "(),{};*!®@[]"
            psm = "7"
        elif t == "con":
            black = "{};*!®@[]"
            psm = "6"
        elif t == "doc":
            black = "{};*!®@[]"
            psm = "3"
        elif t == "inst":
            black = "{};*!®@[]"
            psm = "11"
        if black:
            config = "-c tessedit_char_blacklist={} --psm {} -l rus".format(black, psm)
        elif white:
            config = "-c tessedit_char_whitelist={} --psm {} -l rus".format(white, psm)
        else:
            config = "--psm {} -l rus".format(psm)
        data = pytesseract.image_to_data(allpram[t],
                                         config=config, output_type='data.frame')
        data = data.dropna(0)
        # print(t)
        # print(config)
        # print(data[["text","left","top","width","height","block_num","par_num"]])
        if t == "doc":
            data = data[data["left"] > 10]
            text = " ".join(data[data["block_num"] == data["block_num"].min()]["text"]).strip()
        elif t == "c" or t == "d":
            text = "".join(data["text"].astype(str)).strip()
        else:
            text = " ".join(data["text"].astype(str)).strip()
        if t == "doc" or t == "con":
            res = []
            for tex in text.split():
                if tex.isupper():
                    res.append(tex)
                else:
                    res.append(spell(tex))
            text = " ".join(res)
        stamp_res[t] = text
        if t == "d":
            txt = []
            allpram[t].save("tmp.png")
            date = reader_num.readtext("tmp.png", allowlist="1234567890.")
            for d in date:
                txt.append(d[1])
            text = "".join(txt)
            stamp_res["easy"] = text
        # elif t == "inst":
        #     txt = []
        #     allpram[t].save("tmp.png")
        #     inst = reader.readtext("tmp.png")
        #     for d in inst:
        #         txt.append(d[1])
        #     text = " ".join(txt)
        #     stamp_res["insteasy"] = text
    return stamp_res

def data_preproess(file):
    print(file)
    img_path, list_num = extrstamp_onlytess(file)
    try:
        image = Image.open(img_path)
    except:
        return {}
    width, height = image.size
    newsize = (width*5, height*5)
    im1 = image.resize(newsize)
    if type(list_num) == int:
        fold = file.split("/")[-2]
    else:
        fold = "none_"+file.split("/")[-2]
    os.makedirs("stimg/"+fold, exist_ok=True)
    im1.save("stimg/"+fold+"/"+file.split("/")[-1].replace(".pdf", ".png"))


def allstres(file):
    img_path, list_num = extrstamp_onlytess(file)
    if type(list_num) == int:
        image = Image.open(img_path)
        width, height = image.size
        newsize = (width*5, height*5)
        im1 = image.resize(newsize)
        data = stampDataEextract(im1)
        ret, place = stamp_rasb(data)
        allparam = cropData(im1, ret, place)
        stamp_res = extractCrop(allparam)
        stamp_res["inst"] = fineins(stamp_res)
        stamp_res["easy"] = findedate(stamp_res)

        return stamp_res, list_num
    else:
        return None, None

def stage():
    pass

def ocrlight(path):
    doc = fitz.open(path)
    mat = fitz.Matrix(15, 15)
    filename = ".".join(doc.name.split("/")[-1].split(".")[:-1])
    stamp_res, list_num = allstres(path)

    stage = ""
    designInstitute = ""
    constructionstamp = ""
    docname_stamp = ""
    docdate_stamp = ""
    inventoryNumber_stamp = ""
    change_stamp = ""
    cipher_stamp = ""
    pprint(stamp_res)
    if stamp_res:

        if "s" in stamp_res:
            stage = stamp_res["s"]
        if "doc" in stamp_res:
            docname_stamp = stamp_res["doc"]
        if "con" in stamp_res:
            constructionstamp = stamp_res["con"]
        if "easy" in stamp_res:
            docdate_stamp = stamp_res["easy"]
        if "i" in stamp_res:
            change_stamp = stamp_res["i"]
        if "inst" in stamp_res:
            designInstitute = stamp_res["inst"]
        if "c" in stamp_res:
            cipher_stamp = stamp_res["c"]
    if list_num == 0:
        result = {
            "type": "object",
            "properties": {
                "fileName":

                    { "type": "string", "description": "Имя файла", "value": "" }
                ,
                "documentName":

                    { "type": "string", "description": "Наименование документа", "value": ""}
                ,
                "documentCipher":

                    { "type": "string", "description": "Шифр документа", "value": ""}
                ,
                "constructionName":

                    { "type": "string", "description": "Наименование стройки", "value": "" }
                ,
                "inventoryNumber":

                    { "type": "string", "description": "Инвентарный номер", "value": "" }
                ,
                "milestone":

                    { "type": "string", "description": "Этап", "value": "" }
                ,
                "stage":

                    { "type": "string", "description": "Стадия", "value": stage }
                ,
                "changeNumber":

                    { "type": "string", "description": "Номер изменения", "value": ""}
                ,
                "documentDate":

                    { "type": "string", "description": "Дата документа", "value": "" }
                ,
                "designInstitute":

                    { "type": "string", "description": "Проектный институт", "value": designInstitute }
                ,
                "documentName_stamp":

                    { "type": "string", "description": "Наименование документа", "value": docname_stamp}
                ,
                "documentCipher_stamp":

                    { "type": "string", "description": "Шифр документа", "value": cipher_stamp }
                ,
                "constructionName_stamp":

                    { "type": "string", "description": "Наименование стройки", "value": constructionstamp }
                ,
                "changeNumber_stamp":

                    { "type": "string", "description": "Номер изменения", "value": change_stamp }
                ,
                "documentDate_stamp":

                    { "type": "string", "description": "Дата документа", "value": docdate_stamp }
                ,
            }
        }
        return result
    else:
        for page in doc:
            rect = page.rect  # the page rectangle
            mp = (rect.tl + rect.br) / 2  # its middle point, becomes top-left of clip
            fitz.Point(0, rect.height / 3)
            # print(mp)

            blocks = page.get_text("dict", flags=3)["blocks"]
            sb = []
            fitz_lines = []
            fi = True
            for b in blocks:
                for l in b["lines"]:
                    for line in l["spans"]:
                        sb.append(line)
                        if b"\xef\xbf\xbd"*3 in line["text"].encode("utf-8"):
                            fi = False
                            break
                        fitz_lines.append(line["text"].strip())

            pix = page.get_pixmap(matrix=mat)
            # pix = page.get_pixmap()  # , clip=clip
            # pix = page.get_pixmap()  # render page to an image
            if fi:
                text = "\n".join(fitz_lines)
            else:
                text = ""
            if text == "" or len(text) < 20:
                pix.save("image/tmptitul.jpg")
                data = "image/tmptitul.jpg"
                text = re.sub(r" ", " ", pytesseract.image_to_string(data, config="--psm 6 -l rus")) #\n\b
                text = re.sub("\\n\\n+", "\\n\\n", text)
            else:
                text = re.sub("\\n\\n+", "\\n\\n", text)
            text = dataobjects.remove(text)
            text = dataobjects.fixplace(text)
            splitplace = dataobjects.docsplit(text)
            invent, text, splitplace = dataobjects.inventoryNumber(text, splitplace)
            splitplace = dataobjects.docsplit(text)
            cipher = dataobjects.documentCipher(text)
            docdate = dataobjects.documentDate(text, cipher)
            docend = dataobjects.docEnd(text)
            doctype = dataobjects.docType(text)
            change = dataobjects.changeNumber(text)
            milestoneend = dataobjects.miestoneEnd(text)
            milestone = dataobjects.milestone(text, doctype, milestoneend, splitplace)
            construction = dataobjects.constructionName(text, cipher, doctype, milestoneend, milestone, docend, splitplace)
            docname = dataobjects.documentName(text, cipher, doctype, splitplace, milestone, milestoneend, construction)


            result = {
                "type": "object",
                "properties": {
                    "fileName":

                        { "type": "string", "description": "Имя файла", "value": "" }
                    ,
                    "documentName":

                        { "type": "string", "description": "Наименование документа", "value": re.sub("^(. )?проектная документация|^(‚ )?рабочая документация|^техническая документация|(Ц+|Н+|О+|А+|Е+|И+|Я+){6,}", "", docname["value"], flags=re.IGNORECASE)}
                    ,
                    "documentCipher":

                        { "type": "string", "description": "Шифр документа", "value": cipher["value"] }
                    ,
                    "constructionName":

                        { "type": "string", "description": "Наименование стройки", "value": re.sub("^(. )?проектная документация|^(‚ )?рабочая документация|^техническая документация|(Ц+|Н+|О+|А+|Е+|И+|Я+){6,}", "", construction["value"], flags=re.IGNORECASE) }
                    ,
                    "inventoryNumber":

                        { "type": "string", "description": "Инвентарный номер", "value": invent["value"] }
                    ,
                    "milestone":

                        { "type": "string", "description": "Этап", "value": re.sub("\\( ?(Д|д)оговор.{,80}\\)|(Ц+|Н+|О+|А+|Е+|И+|Я+){6,}", "", milestone["value"]) }
                    ,
                    "stage":

                        { "type": "string", "description": "Стадия", "value": stage }
                    ,
                    "changeNumber":

                        { "type": "string", "description": "Номер изменения", "value": change["value"] }
                    ,
                    "documentDate":

                        { "type": "string", "description": "Дата документа", "value": docdate["value"] }
                    ,
                    "designInstitute":

                        { "type": "string", "description": "Проектный институт", "value": designInstitute }
                    ,
                    "documentName_stamp":

                        { "type": "string", "description": "Наименование документа", "value": docname_stamp}
                    ,
                    "documentCipher_stamp":

                        { "type": "string", "description": "Шифр документа", "value": cipher_stamp }
                    ,
                    "constructionName_stamp":

                        { "type": "string", "description": "Наименование стройки", "value": constructionstamp }
                    ,
                    "changeNumber_stamp":

                        { "type": "string", "description": "Номер изменения", "value": change_stamp }
                    ,
                    "documentDate_stamp":

                        { "type": "string", "description": "Дата документа", "value": docdate_stamp }
                    ,
                }
            }


            return result


# allfileres = {1:{}, 2:{}}
# dodo = [[1,2], [2,3], [3,4], [4,2], [5,2], [6,5], [7,0]]
# for pp, do in zip([1,1,1,1,1,1,1,2,2,2,2,2,2,2,2], [1,2,3,4,5,6,7,1,2,3,4,5,6,7,8]) : #pp, do in zip([1,1,1,1,1,1,1,2,2,2,2,2,2,2,2], [1,2,3,4,5,6,7,1,2,3,4,5,6,7,8])
def ocr(path):
    # doc = fitz.open("template/tmp{}/{}.pdf".format(pp,do))
    doc = fitz.open(path)
    ocr_count = 0
    cp = 0
    titulres = {}
    allres_stamp = []
    for page in doc:
        result_stamp = {'stage': {'easy': "", 'fitz': "", 'tess': ""},
                        'stageSecond': {'easy': "", 'fitz': "", 'tess': ""},
                        'documentDate': {'easy': "",
                                         'fitz': "",
                                         'tess': ""},
                        'documentDateSecond': {'easy': "", 'fitz': "", 'tess': ""},
                        'documentDateThird': {'easy': "", 'fitz': "", 'tess': ""},
                        'changeNumber': {'easy': "",
                                         'fitz': "",
                                         'tess': ""},
                        'designInstitute': {'easy': "",
                                            'fitz': "",
                                            'tess': ""}}
        bb = []
        mat = fitz.Matrix(6, 6)
        rect = page.rect  # the page rectangle
        mp = (rect.tl + rect.br) / 2  # its middle point, becomes top-left of clip
        fitz.Point(0, rect.height / 3)
        # print(mp)

        blocks = page.get_text("dict", flags=0)["blocks"]
        if page.number == 0:
            mp1 = fitz.Point(0, 100)
            mp2 = fitz.Point(rect.width, rect.height - 300)
            clip = fitz.Rect(mp1, mp2)  # the area we want
            # print(clip)
            pix = page.get_pixmap(matrix=mat)  # , clip=clip
            # pix = page.get_pixmap()  # render page to an image
            pix.save("image/page-first-%i.jpg" % page.number)  # store image as a PNG
            titulres = titul_recgnize("image/page-first-%i.jpg" % page.number, tmp1conf["titular"], blocks)
        pix = page.get_pixmap()
        pix.save("image/line.jpg")
        try:
            y2, x2 = linecrop("image/line.jpg")
        except:
            y2, x2 = 0,0

        # print(y2)
        # print(x2)
        # print(rect)
        mediabox = page.mediabox
        w = rect.width - 400
        h = rect.height - 200
        mp1 = fitz.Point(w, h)
        mp2 = fitz.Point(x2 + 1, y2 + 1)
        r = fitz.Rect(0, 0, mediabox.width, x2)
        clip = fitz.Rect(mp1, mp2)  # the area we want

        # print(clip)
        pix = page.get_pixmap(matrix=mat, clip=clip)
        blocks = page.get_text("dict", flags=0, clip=clip)["blocks"]

        filename = "image/page-table-%i.jpg" % page.number

        try:
            pix.save(filename)  # store image as a PNG
            img = cv.imread(cv.samples.findFile(filename))
            cImage = np.copy(img)  # image to draw lines

            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            cv.imwrite(filename, gray)
            easy = reader.readtext(gray)
            # print(easy)
            stageSec = None
            dateSec = None
            for ea in easy:
                if ea[1].lower().strip() == "дата":
                    dateSec = [[ea[0][0][0], ea[0][2][1]],
                               [ea[0][1][0], ea[0][2][1] + (ea[0][2][1] - ea[0][0][1]) + 10]]
                if ea[1].lower().strip() == "стадия":
                    stageSec = [[ea[0][0][0], ea[0][2][1]],
                                [ea[0][1][0], ea[0][2][1] + (ea[0][2][1] - ea[0][0][1]) + 10]]

            for par in tmp1conf["table"]:

                alt_par = None
                if par == "stageSecond" and stageSec is not None:
                    alt_par = stageSec
                elif par == "documentDateSecond" and dateSec is not None:
                    alt_par = dateSec
                if alt_par:
                    tmpar = alt_par
                else:
                    tmpar = tmp1conf["table"][par]["rect"]

                cropped_img = get_cropped_image(gray, tmpar[0][0], tmpar[0][1], tmpar[1][0], tmpar[1][1])
                try:

                    result_stamp[par] = stamp_recognize(cropped_img, tmp1conf["table"][par], par, w, h, alt_par, blocks)

                except Exception as e:
                    print(e)
                cv.imwrite("cropped_img/{}.jpg".format(par), cropped_img)
                # print(par)

        except Exception as e:
            print(e)
        allres_stamp.append(result_stamp)
        cp += 1
        if cp >= 7:
            break
    maxs = 0
    stamp_res = allres_stamp[0]
    for s in allres_stamp:
        size = 0
        for i in s:
            for j in s[i]:
                if s[i][j]:
                    size += 1
                    break
        if maxs < size:
            maxs = size
            stamp_res = s
    pprint({"stamp": stamp_res, "titul": titulres})
    return {"stamp": stamp_res, "titul": titulres}
