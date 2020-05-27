from nltk.lm import Vocabulary
from nltk.lm.models import MLE
from nltk.util import ngrams

N = 3
vocab = Vocablary([word for sent in texts for word in sent], unk_cutoff=2, unl_label='<UNK>')
text_ngrams = [ngrams(sent, N) for sent in texts]
lm = MLE(order=N, vocablary=vocab)
lm.fit(text_ngrams)
