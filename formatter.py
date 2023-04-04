import json
from Levenshtein import distance

with open("available.json", "r") as json_file:
    avail = json.load(json_file)

def best(data):
    try:
        if data["fitz"] != "":
            return data["fitz"]
        elif data["tess"] != "":
            return data["tess"]
        elif data["easy"] != "":
            return data["easy"]
        else:
            return ""
    except Exception as e:
        print(e)
        return ""
def lightmerge(data):
    pass

def merge(data):
    result = {
        "type": "object",
        "properties": {
            "fileName":

                { "type": "string", "description": "Имя файла", "value": "" }
            ,
            "documentName":

                { "type": "string", "description": "Наименование документа", "value": "" }
            ,
            "documentCipher":

                { "type": "string", "description": "Шифр документа", "value": "" }
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

                { "type": "string", "description": "Стадия", "value": "" }
            ,
            "changeNumber":

                { "type": "string", "description": "Номер изменения", "value": "" }
            ,
            "documentDate":

                { "type": "string", "description": "Дата документа", "value": "" }
            ,
            "designInstitute":

                { "type": "string", "description": "Проектный институт", "value": "" }
        }
    }

    stamp = data["stamp"]
    titul = data["titul"]
    documentDate = best(stamp["documentDate"])
    if documentDate == "":
        documentDate = best(stamp["documentDateSecond"])
    if documentDate == "":
        documentDate = best(stamp["documentDateThird"])
    changeNumber = best(stamp["changeNumber"])
    # designInstitute = best(stamp["designInstitute"])
    min = 100
    Institute = ""
    dis = []
    for model in stamp["designInstitute"]:
        di = stamp["designInstitute"][model]
        if di != "":
            dis.append(di)
    for designInstitute in dis:
        for av in avail["designInstitute"]:
            dist = distance(av.lower(), designInstitute.lower())
            if dist < min and dist <= 6:
                min = dist
                Institute = av
    designInstitute = Institute
    if designInstitute == "":
        designInstitute = best(stamp["designInstitute"])
    stage = best(stamp["stage"])
    if stage == "":
        stage = best(stamp["stageSecond"])

    result["properties"]["stage"]["value"] = stage
    result["properties"]["documentDate"]["value"] = documentDate
    result["properties"]["changeNumber"]["value"] = changeNumber
    result["properties"]["designInstitute"]["value"] = designInstitute

    changeNumber = best(titul["changeNumber"])
    if result["properties"]["changeNumber"]["value"] == "":
        result["properties"]["changeNumber"]["value"] = changeNumber
    documentDate = best(titul["documentDate"])
    if result["properties"]["documentDate"]["value"] == "":
        result["properties"]["documentDate"]["value"] = documentDate

    documentCipher = best(titul["documentCipher"])
    constructionName = best(titul["constructionName"])
    documentName = best(titul["documentName"])
    inventoryNumber = best(titul["inventoryNumber"])
    milestone = best(titul["milestone"])

    result["properties"]["documentCipher"]["value"] = documentCipher
    result["properties"]["documentName"]["value"] = documentName
    result["properties"]["constructionName"]["value"] = constructionName
    result["properties"]["inventoryNumber"]["value"] = inventoryNumber
    result["properties"]["milestone"]["value"] = milestone

    return result