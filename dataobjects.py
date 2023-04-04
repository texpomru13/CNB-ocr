import json
import re

with open("config_tmp.json", "r") as json_file:
    param = json.load(json_file)


def stage():
    pass

def docEnd(text):
    result = {"type": "docEnd", "value": "", "line": None, "place": None}
    k = 0
    lines = text.split("\n")
    bre = 10
    ma = 0
    for line in lines:
        match = re.search(param["titular"]["docEnd"]["reg"], line.lower())
        if match:
            # print(line)
            if line[0] == "(":
                for i in range(5):
                    if ")" in lines[k+i]:
                        k = k+i
                        break
            if '"' in line and '"' in lines[k+1]:
                k+=1
            z = k
            # for i in range(1,3):
            #     if lines[z+i] == "":
            #         z+=i

            result = {"type": "docEnd", "value": line, "line": z, "place": 100/len(lines)*k}
        if result["value"]:
            if bre == 0:
                break
            bre -= 1
        k+=1

    return result

def miestoneEnd(text):
    result = {"type": "miestoneEnd", "value": "", "line": None, "place": None}
    k = 0
    lines = text.split("\n")
    for line in lines:
        match = re.search(param["titular"]["miestoneEnd"]["reg"], line.lower())
        if match:
            result = {"type": "miestoneEnd", "value": line, "line": k, "place": 100/len(lines)*k}
            break
        k+=1
    return result

def docType(text):
    result = {"type": "docType", "value": "", "line": None, "place": None}

    lines = text.split("\n")
    k = len(lines)-1
    for line in reversed(lines):
        match = re.search(param["titular"]["documentType"]["reg"], line.lower())
        if match:
            result = {"type": "docType", "value": line, "line": k, "place": 100/len(lines)*k}
        k-=1
    return result

def docsplit(text):
    splitplace = {}
    text = text.strip("\n")
    lines = text.split("\n")
    k = 0
    for line in lines:
        if line == "" and k != 0:
            splitplace[k] = {"type": "split", "value": " ", "line": k, "place": 0}
        k+=1
    return splitplace


def documentDate(text, cipher):
    result = {"type": "documentDate", "value": "", "line": None, "place": None}
    k =0
    minDate = None
    lines = text.split("\n")
    if cipher["place"] and cipher["place"] >40:
        for line in reversed(lines[cipher["line"]:]):
            if result["value"] is None:
                mid = re.search(param["titular"]["minDate"]["reg"], line)
                if mid:
                    span = mid.span()
                    result = {"type": "documentDate", "value": line[span[0]:span[1]], "line": k, "place": 100/len(lines)*k}
            mad = re.search(param["titular"]["maxDate"]["reg"], line)
            if mad:
                span = mad.span()
                result = {"type": "documentDate", "value": line[span[0]:span[1]], "line": k, "place": 100/len(lines)*k}
                break
    else:
        for line in reversed(lines[-int(len(lines)*.2):]):
            if result["value"] is None:
                mid = re.search(param["titular"]["minDate"]["reg"], line)
                if mid:
                    span = mid.span()
                    result = {"type": "documentDate", "value": line[span[0]:span[1]], "line": k, "place": 100/len(lines)*k}
            mad = re.search(param["titular"]["maxDate"]["reg"], line)
            if mad:
                span = mad.span()
                result = {"type": "documentDate", "value": line[span[0]:span[1]], "line": k, "place": 100/len(lines)*k}
                break
    if result["value"]:
        return result
    else:
        for line in lines[:4]:
            if result["value"] is None:
                mid = re.search(param["titular"]["minDate"]["reg"], line)
                if mid:
                    span = mid.span()
                    result = {"type": "documentDate", "value": line[span[0]:span[1]], "line": k, "place": 100/len(lines)*k}
            mad = re.search(param["titular"]["maxDate"]["reg"], line)
            if mad:
                span = mad.span()
                result = {"type": "documentDate", "value": line[span[0]:span[1]], "line": k, "place": 100/len(lines)*k}
                break

    return result


def changeNumber(text):
    result = {"type": "changeNumber", "value": "", "line": None, "place": None}
    k =0
    lines = text.split("\n")
    match = re.search(param["titular"]["changeNumber"]["reg"], text.lower())
    if match:
        span = match.span()
        value = text.lower()[span[0]: span[1]]
        numb = re.findall(r'\d+', value)

        value = "".join(str(i) for i in numb)

        result = {"type": "changeNumber", "value": value, "line": k, "place": 100 / len(lines) * k}
    return result

def designInstitute():
    pass

def remove(text):
    lines = text.split("\n")
    nt = []
    for line in lines:
        line = re.sub("\\S? *\\|\\s*[^a-zA-Zа-яА-Я]?\\s", "", line)
        line = re.sub("\\S? *\\|\\s*[^a-zA-Zа-яА-Я]?\\s", "", line)
        line = re.sub("[°*“|„]", "", line)
        line = re.sub(" +", " ", line).strip()
        match = re.search(param["titular"]["remove"]["reg"], line.lower())
        if match:
            pass
        else:
            nt.append(line)

    return "\n".join(nt)

def fixplace(text):
    nt = []
    k = -1
    lines = text.split("\n")
    notskip = True
    for line in lines:
        k+=1
        if notskip:
            nt.append(line)
        else:
            notskip = True
        try:
            tom = re.search(param["titular"]["tom"]["reg"], line.lower())
            izm = re.search("^изм", lines[k+1].lower())
            tmptext = line+ " " + lines[k+1]
            if tom or izm:
                pass
            else:
                match = re.search(param["titular"]["documentCipher"]["reg"], line + lines[k+1])
                if match and match.end() >= len(line+lines[k+1])-1:
                    nt[-1] += lines[k+1]
                    notskip = False
                    continue
            start = re.search(param["titular"]["documentType"]["reg"], line.lower())
            end = re.search(param["titular"]["documentType"]["reg"], lines[k+1].lower())
            match = re.search(param["titular"]["documentType"]["reg"], tmptext.lower())
            if start or end:
                match = None
            if match:
                nt[-1] += " " +  lines[k+1]
                notskip = False
                continue
            start = re.search(param["titular"]["milestone"]["reg"], line.lower())
            end = re.search(param["titular"]["milestone"]["reg"], lines[k+1].lower())
            match = re.search(param["titular"]["milestone"]["reg"], tmptext.lower())
            if start or end:
                match = None
            if match:
                nt[-1] += " " + lines[k+1]
                notskip = False
                continue
        except:
            pass


    return "\n".join(nt)

def documentCipher(text):
    result = {"type": "Cipher", "value": "", "line": None, "place": None}
    k = 0
    lines = text.split("\n")
    for line in lines:
        match = re.search(param["titular"]["documentCipher"]["reg"], line.replace(" ", "", 1))
        if match:
            result = {"type": "Cipher", "value": line, "line": k, "place": 100/len(lines)*k}
            break

        k+=1
    return result

def documentName(text, cipher, doctype, splitplace, milestone, milestoneend, construction):
    result = []
    k = 0
    text = text.strip("\n")
    lines = text.split("\n")
    end = None
    tend = True
    start = None
    if cipher["value"] == "":
        return {"type": "documentName", "value": "", "line": None, "place": None}
    elif cipher["place"] > 30:

        k = 0
        for line in lines[cipher["line"]:]:
            match = re.search("^изм", line.lower())

            if match:
                end = cipher["line"]+k
                break
            if k > 5:
                break
            k+=1

        k = 0
        if end is None:
            for line in lines[cipher["line"]:]:
                match = re.search(param["titular"]["tom"]["reg"], line.lower())
                if match:
                    end = cipher["line"]+k+1
                    break
                if k > 3:
                    break
                k+=1

        if end is None:
            end = cipher["line"]
    else:
        k = 0
        for line in reversed(lines):
            match = re.search("^изм", line.lower())
            if match:
                end = len(lines)-k-1
                break
            if k > 10:
                break
            k+=1

        k = 0
        if end is None:
            for line in reversed(lines):
                match = re.search(param["titular"]["tom"]["reg"], line.lower())
                if match:
                    end = len(lines)-k
                    break
                if k > 10:
                    break
                k+=1
        if end is None:
            end = len(lines)-2

    if doctype["value"] and doctype["place"] > 18:
        start = doctype["line"]
    elif milestone["value"]:
        start = milestone["end"] +1
    elif milestoneend["value"]:
        start = milestoneend["line"] +1
    elif construction["value"]:
        start = construction["end"] +1
    else:
        return {"type": "documentName", "value": "", "line": None, "place": None}
    res = []
    for l in lines[start:end]:
        if re.search(param["titular"]["documentCipher"]["reg"], l):
            pass
        else:
            res.append(l)
    res = "\n".join(res)
    res = re.sub("\\n", " ", res)
    res = re.sub("\\([^\\(]{69,}\\)|[\\_\\\]", "", res)
    res = re.sub(" +", " ", res)
    return {"type": "documentName", "value": res, "line": start, "place": None}


def constructionName(text, cipher, doctype, milestoneend, milestone, docend, splitplace):
    result = []
    k = 0
    text = text.strip("\n")
    lines = text.split("\n")
    end = None
    tend = True
    start = None
    if docend["value"]:
        end = docend["line"]
        tend = False
    elif doctype["line"] and doctype["place"] < 20 and cipher["place"] < 20:
        end = doctype["line"]
        tend = False
    # elif splitplace:
    #     end = splitplace[0]["line"]+3
    else:
        end = 0
        mi = None
        if splitplace:
            mi = min(splitplace)
            splitplace.pop(mi)
        else:
            mi = 2
        if mi < 2:
            end = mi
        else:
            end = 2

    if milestoneend["line"]:
        start = milestoneend["line"]
    elif milestone["line"]:
        start = milestone["line"] + 1
    elif doctype["line"] and cipher["place"] and doctype["place"] > 18:
        start = doctype["line"]
    else:
        k = 0
        for line in lines:
            match = re.search(param["titular"]["constructionNameLC"]["reg"], line.lower())
            if match:
                start = k
                break
            if k >=12:
                break
            k+=1
        if start is None:
            mi = None
            if splitplace:
                for i in range(5):
                    mim = min(splitplace)
                    splitplace.pop(mim)
                    if mim > end:
                        mi = mim
                        break
                    if splitplace == {}:
                        break
            if mi:
                start = mi
            else:
                start = 2+end
    if tend:
        for i in range(1,5):
            try:
                if lines[start-i] == "":
                    end = start-i
                    break
            except:
                pass
    res = "\n".join(lines[end:start]).strip("\n")
    res = re.sub("\\n", " ", res)
    res = re.sub("[\\_\\\]", "", res)
    res = re.sub(" +", " ", res)
    result = {"type": "constructionName", "value": res, "line": end, "end": start, "place": 100 / len(lines) * k}
    return result

def milestone(text, doctype, miestoneend, splitplace):
    result = []
    k = 0
    lines = text.split("\n")
    if miestoneend["line"]: # and miestoneend["place"] >30
        start = miestoneend["line"]
    else:
        start = 0
    print(start)
    if doctype["line"] and doctype["place"] >30:
        end = doctype["line"]
    else:
        end = len(lines)
    for line in lines[start:end]:
        match = re.search(param["titular"]["milestone"]["reg"], line.lower())
        if match:
            if "»" == line.lower()[match.start()-1]:
                match = None
        if match:
            result.append({"type": "milestone", "value": line, "line": k, "place": 100 / len(lines) * k})
        else:
            try:
                match = re.search(param["titular"]["milestone"]["reg"], line.lower()+lines[k+1].lower())
                if match:
                    result.append({"type": "milestone", "value": line, "line": k, "place": 100 / len(lines) * k})
            except:
                pass
        k+=1

    if result == []:
        result = {"type": "milestone", "value": "", "line": None, "place": None}
    elif end != len(lines) and start != 0:
        result = {"type": "milestone", "value": "\n".join(lines[start:end]), "line": start, "end": end, "place": 100 / len(lines) * start}
    elif end != len(lines):
        result = {"type": "milestone", "value": "\n".join(lines[result[0]["line"]+1:end]), "line": result[0]["line"], "end": end, "place": 100 / len(lines) * result[0]["line"]}
    elif start != 0:
        result = {"type": "milestone", "value": "\n".join(lines[start:result[-1]["line"]]), "line": start, "end": result[-1]["line"], "place": 100 / len(lines) * start}
    else:
        ml = []
        if result[-1]["line"] - 1 in splitplace:
            result = result[:-1]
        k = result[0]["line"]
        last = k
        for res in result:
            if res["line"] - last >1:
                break
            last = res["line"]
            ml.append(res["value"])
        if last + 2 in splitplace:
            ml.append(lines[last+1])
        result = {"type": "milestone", "value": "\n".join(ml), "line": k, "end": len(ml)+k, "place": 100 / len(lines) * k}
    if result["value"]:
        result["value"] = re.sub("(Э|э)(ТАП|тап)", "\nЭтап", result["value"].replace("\n", " ")).strip("\n")
        res = result["value"]
        res = re.sub("\\n", " ", res)
        res = re.sub("[\\_\\\]", "", res)
        result["value"] = re.sub(" +", " ", res)
    return result

def inventoryNumber(text, splitplace):

    result = {"type": "inventoryNumber", "value": "", "line": None, "place": None}

    lines = text.split("\n")
    k = len(lines)
    for line in reversed(lines[-4:]):
        match = re.search(param["titular"]["inventoryNumberUpDwn"]["reg"], line.lower())
        if match:
            result = {"type": "inventoryNumber", "value": line, "line": k, "place": 100 / len(lines) * k}
            break
        k-=1
    if result["value"] is not None:
        return result, text, splitplace
    k = 0
    for line in lines[:4]:
        match = re.search(param["titular"]["inventoryNumberUpDwn"]["reg"], line.lower())
        if match:
            result = {"type": "inventoryNumber", "value": line, "line": k, "place": 100 / len(lines) * k}
            if k + 1 in splitplace:
                text = "\n".join(lines[k+1:])
                splitplace.pop(k+1)
            else:
                text = "\n".join(lines[k:])
            break
        k+=1
    if result["value"] is not None:
        return result, text, splitplace

    k =0
    match = re.search(param["titular"]["inventoryNumber"]["reg"], text.lower())
    if match:
        span = match.span()
        value = text.lower()[span[0]: span[1]]
        numb = re.findall("[\\d\\/]", value)

        value = "".join(str(i) for i in numb)

        result = {"type": "inventoryNumber", "value": value, "line": k, "place": 100 / len(lines) * k}
    return result, text, splitplace