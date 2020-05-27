# source 3.1

import nltk
import spacy
nlp = spacy.load('ja_ginza')

from pathlib import Path
Dpath = 'Resources\\DBDC\\'
pathlist = []
pathlist.append(Path(Dpath+'projectnextnlp\\json\\init100\\'))
pathlist.append(Path(Dpath+'projectnextnlp\\json\\rest1046\\'))

import json

texts = []
for datapath in pathlist:
    for file in datapath.glob('*.json'):
        with open(file, encoding ='utf-8') as f:
            json_data = json.load(f)
        turns = json_data['turns']
        for t in turns:
            doc = nlp(t['utterance'])
            for sent in doc.sents:
            texts.append(['<s>'] + [token.orth_ for token in sent] + ['</s>'])
            
# source 3.2

from nltk.lm import Vocabulary
from nltk.lm.models import MLE
from nltk.util import ngrams

N = 3
vocab = Vocablary([word for sent in texts for word in sent], unk_cutoff=2, unl_label='<UNK>')
text_ngrams = [ngrams(sent, N) for sent in texts]
lm = MLE(order=N, vocablary=vocab)
lm.fit(text_ngrams)

# source 3.3

context = ['好きな', 'な'] # 文脈の作成
print(context, '->')

prob_list = [[lm.source(w, context), for w in lm.context_counts(lm.vocab.lookup(context))]
prob_list.sort(reverse = True) # 出現確率順にソート

for prob, word in prob_list[:10]:
    print('\t{:s}: {:f}'.format(word, prob))

# >> ['好き', 'な'] ->
# >> ん: 0.296943
# >> の: 0.183406
# >> 人: 0.109170
# >> 食べ物: 0.100437
# >> もの: 0.087336
# >> 料理: 0.021834
# >> こと: 0.017467
# >> スポーツ: 0.013100
# >> 飲み物: 0.008734
# >> 音楽: 0.008734             
             
# source 3.4

print(lm.generate(text_seed=context , num_words =3))
             
# >> ['食べ物', 'は', '何'] 
