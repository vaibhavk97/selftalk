## Unsupervised Commonsense Question Answering with Self-Talk

This repository contains the code used in the paper:

**Unsupervised Commonsense Question Answering with Self-Talk** 

*Vered Shwartz, Peter West, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi*. EMNLP 2020. [link](https://arxiv.org/abs/2004.05483)


This is a generic framework for incorporating relevant background knowledge into unsupervised models for multiple choice common sense reasoning tasks. The knowledge comes either from:

1) External resources such as ConceptNet; or 
2) Generated by language models in a process of asking clarification questions (means) and using the answers (goal) to clarify the instance. 

### Bug fix - July 2021 (tagged V1.1):

- Using `eos_token` as the padding token in LMs that don't have a padding token (instead of 0). The results are now better for all models, but the general trend remains the same. See [issue #1](https://github.com/vered1986/self_talk/issues/1).


### Tasks
 
The `data` directory contains the `dev.jsonl` and `test.jsonl` of the following tasks:

#### COPA

[Gordon et al. (2012)](https://www.aclweb.org/anthology/S12-1052/): Choices of Plausible Alternatives. Each instance consists of a premise, one of two question types - what caused it or what was the effect, and two choices. 

#### SocialIQA

[Sap et al. (2019)](https://www.aclweb.org/anthology/D19-1454/): Social Interaction Question Answering. Multiple choice questions regarding social interactions. Each instance consists of a context, question and choices. 

#### CommonsenseQA

[Talmor et al. (2019)](https://www.aclweb.org/anthology/N19-1421/): Multiple choice questions around a target concept with 5 choices, each somehow related to the concept but only one is correct. 

#### MCTaco

[Zhou et al. (2019)](https://www.aclweb.org/anthology/D19-1332/): Multiple choice questions about temporal apsects. 

#### PIQA

[Bisk et al. (2020)](https://arxiv.org/abs/1911.11641): Physical Interaction Question Answering. Multiple choice questions regarding physical interactions. Each instance consists of a goal and two alternative solutions. 

#### Winogrande

[Sakaguchi et al. (2020)](https://arxiv.org/pdf/1907.10641.pdf): A large-scale dataset for the Winograd Schema Challenge (WSC) that exhibits less bias and on which models perform substantially worse than humans (it was adversarially filtered to remove word associations and easy examples). Each context discusses two concepts/entities and contains a placeholder into which only one of those entities can be placed. 


### Generating the clarifications

Before you start, make sure you've installed all the requirements in `requirements.txt`. 

`bash generate_all_lm_clarifications.sh [dataset]` will generate all the self-talk clarifications for a specific dataset. 
It assumes an 8 GPU machine and utilizes all the available GPUs (one GPU per process).

For the external resources:

1. ConceptNet: run the following script for each dataset:

```
usage: generate_clarifications_from_conceptnet.py [-h] --dataset DATASET
                                                  [--dataset_type DATASET_TYPE]
                                                  --out_file OUT_FILE
                                                  [--answer_redundancy ANSWER_REDUNDANCY]
                                                  [--max_clarifications MAX_CLARIFICATIONS]
                                                  [--max_length MAX_LENGTH]
                                                  [--conceptnet_dir CONCEPTNET_DIR]

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     Jsonl file
  --dataset_type DATASET_TYPE
                        base dataset format (winogrande, socialiqa,
                        commonsenseqa, mctaco, piqa, or copa)
  --out_file OUT_FILE   Output jsonl file
  --answer_redundancy ANSWER_REDUNDANCY
                        how many answers to generate from each question
  --max_clarifications MAX_CLARIFICATIONS
                        how many clarifications to keep
  --max_length MAX_LENGTH
                        maximum path length in edges
  --conceptnet_dir CONCEPTNET_DIR
                        ConceptNet directory
```

In the first run, it will download and process the ConceptNet data, and save it in `CONCEPTNET_DIR`.

2. COMET: 

Make sure you've installed the Comet reimplementation from [here](https://github.com/vered1986/comet-commonsense). 

Run the following script for each dataset:

```
usage: generate_clarifications_from_comet.py [-h] --dataset DATASET
                                             [--dataset_type DATASET_TYPE]
                                             --out_file OUT_FILE
                                             [--device DEVICE]
                                             [--model_file MODEL_FILE]

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     Jsonl file
  --dataset_type DATASET_TYPE
                        base dataset format (winogrande, socialiqa,
                        commonsenseqa, mctaco, piqa, or copa)
  --out_file OUT_FILE   Output jsonl file
  --device DEVICE       cpu or GPU device
  --model_file MODEL_FILE
                        The COMET pre-trained model
```

3. Google Ngrams: run the following script for each dataset:

```
usage: generate_clarifications_from_googlengrams.py [-h] --dataset DATASET
                                                    [--dataset_type DATASET_TYPE]
                                                    --out_file OUT_FILE
                                                    [--answer_redundancy ANSWER_REDUNDANCY]
                                                    [--max_clarifications MAX_CLARIFICATIONS]
                                                    [--min_freq MIN_FREQ]

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     Jsonl file
  --dataset_type DATASET_TYPE
                        base dataset format (winogrande, socialiqa,
                        commonsenseqa, mctaco, or copa)
  --out_file OUT_FILE   Output jsonl file
  --answer_redundancy ANSWER_REDUNDANCY
                        how many answers to generate from each question
  --max_clarifications MAX_CLARIFICATIONS
                        how many clarifications to keep
  --min_freq MIN_FREQ   minimum co-occurrence frequency to consider

```

Notice that the script assumes a Google Ngrams directory processed as in [here](https://github.com/vered1986/PythonUtils/tree/master/corpora/google_ngrams).

If you want to download the pre-computed clarifications, you need to install [Git LFS](https://git-lfs.github.com/) and run:
```bash
git lfs install
git lfs pull
```

### Model

To compute the baseline results (based on LM score without clarifications), run `bash experiments/[dataset]/predict_baseline_zero_shot.sh [device] dev`, replacing `[device]` by a GPU number. It will save a table with all the results under `output/[dataset]/baseline_zero_shot_results.tsv` and a prediction file under `data/[dataset]/dev_predictions.jsonl`.

To compute the model results, run `bash experiments/[dataset]/predict_zero_shot.sh [device] dev`, replacing `[device]` by a GPU number. It will save a table with all the results under `output/[dataset]/zero_shot_results.tsv` and a prediction file for each knowledge source under `data/[dataset]/dev_clarified_[knowledge_source]_predictions.jsonl`. 

### To Add a New Task

1. Add the a new directory under `data` with the files `train.jsonl`, `dev.jsonl`, and `test.jsonl`.
2. Add a new `InstanceReader` in `multiple_choice_asking_questions_predictor_unsupervised.py` and in `multiple_choice_baseline_predictor_unsupervised.py`.
3. Change the clarification generator scripts with dataset-specific clarification generation. 
4. Copy the content under `experiments/[dataset]`, change the dataset name in the scripts and configure the question and answer prefixes in `prefixes.json`.


#### References 

Please cite this repository using the following reference:

```
@inproceedings{self_talk_2020,
  title={Unsupervised Commonsense Question Answering with Self-Talk},
  author={Vered Shwartz and Peter West and Ronan Le Bras and Chandra Bhagavatula and and Yejin Choi},
  booktitle={EMNLP},
  year={2020}
}
```
