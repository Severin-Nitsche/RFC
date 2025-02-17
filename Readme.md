# Public Opinion and Awareness on Matters of Privacy
This repository contains my Code for the *research focus class - medical data privacy*.

## Focus
In this project, I aim to support previous findings regarding privacy online using data-centric methods.
Particularly, [social studies](https://journals.sagepub.com/doi/abs/10.1177/009365021141833) have shown that people tend to be pretty careless when it comes to privacy online *even after* they experienced negative consequences from that.

I aim to support this finding with a data-centric approach. To this end, I have three research questions:

1. What attitudes do redditors express towards privacy?
2. How much private data do redditors share?
3. Is there a relation between a redditors attitude towards privacy and their data shareage?

## Study Design

I want to conduct a correlational study, where I examine the variables opinion on privacy and sharing of private information online. 

I operationalize the variable regarding the opinion, by analyzing reddit posts using *category based sentiment analysis*. The mode of this variable is self report, as I take the participants written word for their opinion. The scale is ordinal and will be represented as a weighted average of the categories negative, neutral and positive across all applicable posts of the participant.

I operationalize the second variable regarding the leakage of private information, by analyzing reddit posts using machine learning, to detect which parts of a post are considerably private, meaning they regard one of the categories *belief, politics, sex, ethnic or health*. The found instances of data-leakage will then be aggregated into a score for this participant, which makes this variable again ordinal scaled. The mode is behavioral.

Regarding internal validity, this study is pretty low, as for one, only automized methods are used for analyzing and also, this being a correlational study also does not help.

For external validity, threats are the selection of redditors to analyze, as redditors active in other subreddits and thus different communities might exhibit different characteristics. However, due to the automized nature, the number of participants which I aim to be a thousand is pretty high and can be exactly repeated which makes this actually pretty strong.

Now, considering ethics, I do not ask participant, i. e. redditors for their consent to analyze their data which must be seen critically. However the post data will not be stored longer than needed for analysis and in the end, the datapoints will not be traceable to a specific account. Also, the participants voluntarily shared their data online, where they have to expect it being seen by anyone. Further, due to the non-invasive nature of this study, the participants will not be strained at all and thus not experience more strain than daily life.

## Organization
To analyze the reddit posts, I wrote a scraper that saves some posts in a `.json` file. The code can be found in `reddit.py`.

To conduct the sentiment analysis, I used [PyABSA](https://github.com/yangheng95/PyABSA). This code can be found in `sentiment.py` uses the `.json` file produced with the scraper and amends sentiment information to it.

To analyze the disclosure of private information, I used a longformer-based method from [The Text Anonymization Benchmark (TAB)](https://arxiv.org/abs/2202.00443), whoms repository I forked and amended [here](https://github.com/Severin-Nitsche/text-anonymization-benchmark). There is also information to find on how to run this on the RWTH Cluster, as training a model naturally takes a lot of resources.

As the training data for the TAB is not tailored specifically towards reddit and sanity checks have shown that the perfomance is not acceptable, I have set up Label studio, setup in `label-studio.xml`, to be able to annotate the data manually and use as training data.
But I cannot bring the kind of man power for this kind of work needed, as I am only a one man army. Hence, I'll set DeepSeek V3 on that task using the approach devised in [GPT-NER](https://arxiv.org/pdf/2304.10428) to annotate the data.

To conduct the LLM-labelling, I'll write a script that can be found in `deepseek.py`.

## Reproducing
I try to be as reproducible as possible, by conducting my experiments in an isolated `nix-shell` and a python `venv`. I also try to maintain an up to date copy of `requirements.txt`.

## To Do
- [x] Save the Label Studio configuration to git
- [ ] Aggregate the sentiment data
- [ ] Implement GPT-NER
  - [ ] Get DeepSeek V3 running
  - [ ] Get output format enforcing working
  - [ ] Retrain TAB
- [ ] Build final pipeline
