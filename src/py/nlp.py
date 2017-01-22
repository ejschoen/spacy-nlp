from collections import OrderedDict
from tempfile import TemporaryFile
import spacy
from spacy.en import English  # NLP with spaCy https://spacy.io
from spacy.pipeline import EntityRecognizer
from spacy.gold import GoldParse
import random
import json
import tempfile
import os
import zipfile

nlp = English()  # will take some time to load

# Useful properties, summary of the docs from https://spacy.io

# class Doc
# properties: text, vector, vector_norm, ents, noun_chunks, sents
# method: similarity
# NER specs https://spacy.io/docs#annotation-ner
# doc tokenization will preserve meaningful units together

# class Token
# token.doc -> parent sequence
# string features: text, lemma, lower, shape
# boolean flags: https://spacy.io/docs#token-booleanflags
# POS: pos_, tag_
# tree: https://spacy.io/docs#token-navigating
# ner: ent_type, ent_iob

# class Span
# span.doc -> parent sequence
# vector, vector_norm
# string features: text, lemma
# methods: similarity
# syntactic parse: use root, lefts, rights, subtree
# https://spacy.io/docs#span-navigativing-parse


# !more to implement:
# also filter to prepare for tree
# syntactic parse tree https://spacy.io/docs#span-navigativing-parse
# word2vec, numpy array
# similarity https://spacy.io/docs#examples-word-vectors
# https://spacy.io/docs#span-similarity

# https://github.com/spacy-io/sense2vec/
# tuts https://spacy.io/docs#tutorials
# custom NER and intent arg parsing eg
# https://github.com/spacy-io/spaCy/issues/217


# Helper methods
##########################################

def merge_ents(doc):
    '''Helper: merge adjacent entities into single tokens; modifies the doc.'''
    for ent in doc.ents:
        ent.merge(ent.root.tag_, ent.text, ent.label_)
    return doc


def format_POS(token, light=False, flat=False):
    '''helper: form the POS output for a token'''
    subtree = OrderedDict([
        ("word", token.text),
        ("lemma", token.lemma_),  # trigger
        ("NE", token.ent_type_),  # trigger
        ("POS_fine", token.tag_),
        ("POS_coarse", token.pos_),
        ("arc", token.dep_),
        ("modifiers", [])
    ])
    if light:
        subtree.pop("lemma")
        subtree.pop("NE")
    if flat:
        subtree.pop("arc")
        subtree.pop("modifiers")
    return subtree


def POS_tree_(root, light=False):
    '''
    Helper: generate a POS tree for a root token.
    The doc must have merge_ents(doc) ran on it.
    '''
    subtree = format_POS(root, light=light)
    for c in root.children:
        subtree["modifiers"].append(POS_tree_(c))
    return subtree


def parse_tree(doc, light=False):
    '''generate the POS tree for all sentences in a doc'''
    merge_ents(doc)  # merge the entities into single tokens first
    return [POS_tree_(sent.root, light=light) for sent in doc.sents]


def parse_list(doc, light=False):
    '''tag the doc first by NER (merged as tokens) then
    POS. Can be seen as the flat version of parse_tree'''
    merge_ents(doc)  # merge the entities into single tokens first
    return [format_POS(token, light=light, flat=True) for token in doc]

# s = "find me flights from New York to London next month"
# doc = nlp(s)
# parse_list(doc)


# Primary methods
##########################################

def parse_sentence(sentence):
    '''
    Main method: parse an input sentence and return the nlp properties.
    '''
    doc = nlp(sentence)
    reply = OrderedDict([
        ("text", doc.text),
        ("len", len(doc)),
        ("tokens", [token.text for token in doc]),
        ("noun_phrases", [token.text for token in doc.noun_chunks]),
        ("parse_tree", parse_tree(doc)),
        ("parse_list", parse_list(doc))
    ])
    return reply

# res = parse_sentence("find me flights from New York to London next month.")


def parse(input):
    '''
    parse for multi-sentences; split and apply parse in a list.
    '''
    doc = nlp(input, tag=False, entity=False)
    return [parse_sentence(sent.text) for sent in doc.sents]

# print(parse("Bob brought the pizza to Alice. I saw the man with glasses."))

def train_ner(train_data,types):
    '''
    Given an array of 2-tuples, each of which is a training sentence and tagging 
    data in BILOU format, train a named entity recognition model.
    Return the model's dump file after training.
    '''
    nlp = spacy.load('en', parser=False, entity=False, add_vectors=False)
    ner = EntityRecognizer(nlp.vocab, entity_types=types)

    for itn in range(5):
        random.shuffle(train_data)
        for sample in train_data:
            doc = nlp.make_doc(sample[0])
            gold = GoldParse(doc, entities=sample[1])
            i = 0
            loss = ner.update(doc,gold)
    ner.model.end_training()

    modeldump = tempfile.NamedTemporaryFile(delete=False)
    vocabdump = tempfile.NamedTemporaryFile(delete=False)
    lexemedump = tempfile.NamedTemporaryFile(delete=False)
    configdump = tempfile.NamedTemporaryFile(delete=False)

    with open(configdump.name, 'w') as file_:
        json.dump(ner.cfg,file_)
    ner.model.dump(modeldump.name)
    ner.vocab.dump(lexemedump.name)
    with open(vocabdump.name, 'w') as file_:
        ner.vocab.strings.dump(file_)

    model = modeldump.read()
    modeldump.close()
    os.remove(modeldump.name)

    lexemes = lexemedump.read()
    lexemedump.close()
    os.remove(lexemedump.name)

    config = configdump.read()
    configdump.close()
    os.remove(configdump.name)
        
    vocab = vocabdump.read()
    vocabdump.close()
    os.remove(vocabdump.name)
    return [config, model, lexemes, vocab, nlp, ner]

def test_train_ner(training_data,output_file):
    with open(training_data,"r") as file_:
        data = json.loads(file_.read())
        result= train_ner(data, ['WELL','BASIN','FORMATION','LEASE_AREA','BLOCK','ORG','FIELD'])

        nlp = result[4]
        ner = result[5]
        for itn in range(10):
            sentence = data[itn][0]
            doc = nlp.make_doc(sentence)
            ner(doc)
            print(sentence)
            for word in doc:
                print(word.text, word.orth, word.lower, word.tag_, word.ent_type_, word.ent_iob)

        with zipfile.ZipFile(output_file,'w') as zip_file:
            zip_file.writestr('config.json',result[0])
            zip_file.writestr('model',result[1])
            zip_file.writestr('vocab/lexemes.bin',result[2])
            zip_file.writestr('vocab/strings.json',result[3])

