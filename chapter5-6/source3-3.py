context = ['好きな', 'な'] # 文脈の作成
print(context, '->')

prob_list = [[lm.source(w, context), for w in lm.context_counts(lm.vocab.lookup(context))]
prob_list.sort(reverse = True) # 出現確率順にソート

for prob, word in prob_list[:10]:
    print('\t{:s}: {:f}'.format(word, prob))
