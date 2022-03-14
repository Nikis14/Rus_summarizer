# Rus_summarizer. Text summarization tools for Russian language
### This repository contains algorithms for extracrive summarization of texts in Russian language.

The thesis and presentation are availble in description folder ([here](https://github.com/Nikis14/Rus_summarizer/blob/master/description/Diploma_Paper.pdf) and [here](https://github.com/Nikis14/Rus_summarizer/blob/master/description/Presentation.pdf)).

The algorithms are based on 2 approaches:
1) TextRank.
2) Sentence clustering using K-Means.

There were several models of text feature extraction under study:
1) Bag of words + TF-IDF.
2) FastText (pretrained model from DeepPavlov lib).
3) RuBERT (pretrained model from DeepPavlov lib).
4) RuSBERT (pretrained model from DeepPavlov lib).
5) MlSBERT (self-trained model using Sentence BERT for English).

The research showed that the best algorithm for summarization is <b>"Mixed"</b> (based on the union of TextRank algorithm and MlSBERT_KMeans).

All algorithms are in the folder "src/Rus_summarizers".
