# PMC paper Supplementary Material

This repository contains both code and name of the datasets' files used to perform the experiments presented in the paper entitled **"Transfer Learning for the efficient Detection of COVID-19 from Smartphone Audio Data"**, submitted to the Elsevier Pervasive and Mobile Computing (PMC) journal.

# Introduction
Disease detection from smartphone data represents an open research challenge in m-health systems. COVID-19 and its respiratory symptoms are an important case study in this area and a potential real instrument to counteract the pandemic situation.
The efficacy of this solution mainly depends on the performances of AI algorithms applied to the collected data and their possible implementation directly on the users' mobile devices.
Considering these issues, and the limited amount of available data, in this paper we present the experimental evaluation of 3 different deep learning models, compared also with hand-crafted features, and of two main approaches of transfer learning in this scenario: *feature extraction* and *fine-tuning*. Specifically, we considered **VGGish**, **YAMNET**, and **L3-Net** (including 12 different configurations) evaluated through user-independent experiments on 4 different datasets (13,447 samples in total).

Results clearly show the advantages of L3-Net in all the experimental settings as it overcomes the other solutions by 12.3% in terms of Precision-Recall AUC as features extractor, and by 10% when the model is fine-tuned.
Moreover, we note that fine-tuning only the fully-connected layers of the pre-trained models generally leads to worse performances, with an average drop of 6.6% with respect to feature extraction.
Finally, we evaluate the memory footprints of the different models for their possible applications on commercial mobile devices. 
