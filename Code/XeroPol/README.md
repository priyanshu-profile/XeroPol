# XeroPol: Zero-shot cross-lingual transformer alignmentXeroPol:Emotion-aware Contrastive Learning for Zero-shot Cross-lingual Politeness Identification in Dialogues

## Dependencies

- python>=3.8
- torch==1.11.0
- transformers==4.20.0
- sentencepiece==0.1.96


#### Datasets

We used two goal-oriented datasets in this work: 1. Politeness-annotated MultiDoGo 2. EmoWOZ

The sample data is provided in data folder: 
	Raw data as multidogo.csv and emowoz.csv
	Processed data is available in data/multidogo and data/emowoz subfolders.

The preprocessing code in **`preprocess.py`** : This will generate the required files and subdirectories.
 
Politeness labels: {0: impolite, 1: somewhat_impolite, 2: somewhat_polite, 3: polite}
Sentiment labels: {0: neutral, 1: negative, 2: positive}																		

#### Pretrained Transformers

The XLM-R pretrained model(s) needs to be downloaded. It can be downloaded from **HuggingFace**. The base XLM-R can be downloaded [here](https://huggingface.co/xlm-roberta-base/tree/main) and the large model [here](https://huggingface.co/xlm-roberta-large/tree/main). 

#### Running Experiments

In the **`config`** folder, most setups as shell files that were reported in the paper are saved. Here is an example:

Open the command line and type: **`nohup ./config/emowoz.sh emowoz_aligned base &`** this command will run the **`emowoz_aligned`** experiments with XLM-R Base. 

Once you trained an English model for EmoWOZ, for instance, type: **`nohup ./config/emowoz.sh emowoz_zero_shot base &`**. This will give the baseline zero-shot scores for EmoWOZ for XLM-R base.

Finally, **`nohup ./config/emowoz.sh emowoz_target base &`** should train the base XLM-R on the labeled data, referred to as 'Target' in the paper.

### Dataset Courtsey

**Politeness-annotated MultiDoGo Dataset**

@article{mishra2022please,
  title={Please be polite: Towards building a politeness adaptive dialogue system for goal-oriented conversations},
  author={Mishra, Kshitij and Firdaus, Mauajama and Ekbal, Asif},
  journal={Neurocomputing},
  volume={494},
  pages={242--254},
  year={2022},
  publisher={Elsevier}
}

**EmoWOZ Dataset**

@inproceedings{feng-etal-2022-emowoz,
      title={EmoWOZ: A Large-Scale Corpus and Labelling Scheme for Emotion Recognition in Task-Oriented Dialogue Systems}, 
      author={Shutong Feng and Nurul Lubis and Christian Geishauser and Hsien-chin Lin and Michael Heck and Carel van Niekerk and Milica GaÅ¡iÄ‡},
      booktitle = "Proceedings of the 13th Language Resources and Evaluation Conference",
      month = June,
      year = "2022",
      address = "Marseille, France",
      publisher = "European Language Resources Association",
      language = "English",
      ISBN = "979-10-95546-72-6",
}
