import nltk
import spacy
nlp = spacy.load('ja_ginza')

from pathlib import Path
Dpath = 'Resources\\DBDC\\'
pathlist = []
pathlist.append(Path(Dpath + 'projectnextnlp\\json\\init100\\'))
pathlist.append(Path(Dpath + 'projectnextnlp\\json\\rest1046\\'))

import json

texts = []
for datapath in pathlist:
    for file in datapath.glob('*.json'):
        with open(file , encoding ='utf-8')as f:
            json_data = json.load(f)
        turns = json_data['turns']
        for t in turns:
            doc = nlp(t['utterance'])
            for sent in doc.sents:
            texts.append(['<s>'] + [token.orth_ for token in sent] + ['</s>'])
