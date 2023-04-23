# Databricks notebook source
# MAGIC %pip install datasets

# COMMAND ----------

BASE = "/dbfs/FileStore/kian/project/datasets/2014 - Deidentification & Heart Disease/"

from spacy.tokens import Doc
from spacy.training import Example
import spacy
from xml.etree import ElementTree as et
nlp = spacy.blank("en")

# COMMAND ----------

import tarfile
import os
import re

def _read_tar_gz( fpath: str) -> dict:
    """
    Read .tar.gz file
    """
    # Open tar file
    tf = tarfile.open(fpath, "r:gz")

    for tf_member in tf.getmembers():
        file_object = tf.extractfile(tf_member)
        name = tf_member.name
        file_name = os.path.basename(name).split(".")[0]
        if re.search(r"\.xml", name) is not None:
            xml_flag = True
        else:
            xml_flag = False
        yield {
            "file_object": file_object,
            "file_name": file_name,
            "xml_flag": xml_flag,
        }
        
from spacy.tokens import Span
        
def _read_task1_file(file_object, file_name):
    xmldoc = et.parse(file_object).getroot()
    entities = xmldoc.findall("TAGS")[0]
    text = xmldoc.findall("TEXT")[0].text
    phi = []
    for entity in entities:
        phi.append(
            {
                "id": entity.attrib["id"],
                "offsets": [[entity.attrib["start"], entity.attrib["end"]]],
                "type": entity.attrib["TYPE"],
                "text": [entity.attrib["text"]],
                "comment": entity.attrib["comment"],
                "normalized": [],
            }
        )

    #document = {"document_id": file_name, "text": text, "phi": phi}
    doc = nlp(text)
    #doc._.document_id = file_name
    res = [ (*p["offsets"][0], p["type"]) for p in phi ]
    ents = [ (int(r[0]), int(r[1]), r[2]) for r in res]
    res = [ doc.char_span(int(r[0]), int(r[1]), label=r[2], alignment_mode="expand") for r in res ]
    
    final = []
    prev_end = -1
    for r in res:
        if r.start > prev_end:
            final.append(r)
            prev_end = r.end
    doc.set_ents(final)
    doc = doc[:510].as_doc()
    return doc, ents

# COMMAND ----------

trainfile = "training-PHI-Gold-Set2.tar.gz"
testfile = "testing-PHI-Gold-fixed.tar.gz"
def get_all_docs(filename):
    examples = []
    ents = []
    for i, document in enumerate(_read_tar_gz(BASE + filename)):
        if document["xml_flag"]:
            example, ent = _read_task1_file(document["file_object"], document["file_name"])
            examples.append(example)
            ents.append(ent)
    return examples, ents
train, train_ents = get_all_docs(trainfile)
test, test_ents = get_all_docs(testfile)

# COMMAND ----------

from spacy.tokens import DocBin
root = "/dbfs/FileStore/kian/project/spacy_datasets/"

db = DocBin()
for d in train:
    db.add(d)
db.to_disk("train_med.spacy")
db.to_disk(root + "train_med.spacy")
    
db = DocBin()
for d in test:
    db.add(d)
db.to_disk("test_med.spacy")
db.to_disk(root+"test_med.spacy")

# COMMAND ----------

from spacy import displacy
displayHTML(displacy.render(train[0], style="ent"))

# COMMAND ----------

db = DocBin().from_disk(path="train_med.spacy")

# COMMAND ----------

from datasets import load_dataset

dataset = load_dataset("conll2003")

# COMMAND ----------

train = dataset["train"].to_pandas()

# COMMAND ----------

nlp

# COMMAND ----------

df = dataset.to_pandas()

# COMMAND ----------

def proc_row(row):
    text = " ".join(row["tokens"])
    phi = row["ner_tags"]
    doc = nlp(text)
    #doc._.document_id = file_name
    res = [ (*p["offsets"][0], p["type"]) for p in phi ]
    ents = [ (int(r[0]), int(r[1]), r[2]) for r in res]
    res = [ Span(int(r[0]), int(r[1]), label=r[2]) for r in res ]
    doc.set_ents(res)
    return doc

# COMMAND ----------

row = train.to_dict("records")[0]

# COMMAND ----------

proc_row(row)

# COMMAND ----------


