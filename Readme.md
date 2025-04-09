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

The `src` folder contains several modules for the different tasks in the pipeline.
Most tasks have a corresponding requirements file in the `requirements` folder.
The intended workflow for using the pipeline is to
1. Load the required python environment for the task.
*Some tasks need a different python version than others.*
2. Load / create the corresponding `venv`.
*Possibly install the requirements file with `pip`.*
3. Run the required Module

Across all modules, the `src/config.py` serves as the configuration file where various parameters such as file paths and other things are defined.

### Reddit Scraper
To analyze the reddit posts, I wrote a scraper that saves some posts in a `.json` file. The code can be found in `reddit.py`.

#### Load the environment
<dl>
<dt>With <code>nix</code></dt>
<dd>

```
nix develop
```

</dd>
<dt>With <code>module</code></dt>
<dd>

```
ml load GCCcore/.12.2.0
ml load Python/3.10.8
```

</dd>
</dl>

#### Create the virtual environment
```
python -m venv venv/reddit
source venv/reddit/bin/activate
pip install -r requirements/reddit.txt
deactivate
```

#### Execute the scraper
```
source venv/reddit/bin/activate
python -m src.reddit
deactivate
```

### Sentiment Analysis
To conduct the sentiment analysis, I used [PyABSA](https://github.com/yangheng95/PyABSA). This code can be found in `sentiment.py` uses the `.json` file produced with the scraper and amends sentiment information to it.

#### Load the environment
<dl>
<dt>With <code>nix</code></dt>
<dd>

```
nix develop
```

</dd>
<dt>With <code>module</code></dt>
<dd>

```
ml load GCCcore/.12.2.0
ml load Python/3.10.8
```

</dd>
</dl>

#### Create the virtual environment
```
python -m venv venv/sentiment
source venv/sentiment/bin/activate
pip install -r requirements/sentiment.txt
deactivate
```

#### Execute the analysis
```
source venv/sentiment/bin/activate
python -m src.sentiment
deactivate
```

### Privacy Leakage Detection
To analyze the disclosure of private information, I used a longformer-based method from [The Text Anonymization Benchmark (TAB)](https://arxiv.org/abs/2202.00443), whoms repository I forked and amended [here](https://github.com/Severin-Nitsche/text-anonymization-benchmark). There is also information to find on how to run this on the RWTH Cluster, as training a model naturally takes a lot of resources.

As the training data for the TAB is not tailored specifically towards reddit and sanity checks have shown that the perfomance is not acceptable, I have set up Label studio, setup in `label-studio.xml`, to be able to annotate the data manually and use as training data.
But I cannot bring the kind of man power for this kind of work needed, as I am only a one man army. Hence, I'll set DeepSeek V3 on that task using the approach devised in [GPT-NER](https://arxiv.org/pdf/2304.10428) to annotate the data.

To conduct the LLM-labelling, I'll write a script that can be found in `deepseek/`.

#### Load the environment for TAB
<dl>
<dt>With <code>nix</code></dt>
<dd>

```
nix develop
```

</dd>
<dt>With <code>module</code></dt>
<dd>

```
ml load GCCcore/.12.2.0
ml load Python/3.10.8
```
*Note that I strongly advise you to run this on the cluster, unless you have a powerful GPU*
</dd>
</dl>

#### Create the virtual environment for TAB
```
python -m venv venv/tab
source venv/tab/bin/activate
pip install -r requirements/sentiment.txt
deactivate
```

#### Train the TAB
```
source venv/tab/bin/activate
python -m src.tab.train_model
deactivate
```
*Note that this training expects some data for fine tuning under `config.CONVERTED_POSTS`.
If you just want to try the training, comment the respective fine tuning loop + data loading.*

If you are a user of CLAIX, you may also use the provided script:
```
sbatch tab.zsh
```
*Note that you may augment the script to run the analysis if you want.*

#### Use the TAB
```
source venv/tab/bin/activate
python -m src.tab.toy_example
deactivate
```
*Note that this uses the scraped `posts.json`.
It also outputs a new `.json`.
If you want some nice terminal output,
there is a pretty print hidden in `data_manipulation.py`.
You can view an example of this in the commit history.*

#### Load the environment for Label Studio
```
nix develop .#label-studio
```

#### Create the virtual environment for Label Studio
```
python -m venv venv/label-studio
source venv/label-studio/bin/activate
pip install -r requirements/label-studio.txt
deactivate
```

#### Open the studio
```
source venv/label-studio/bin/activate
label-studio
```

#### Work with the studio
1. Create an account and log in
2. Create a new project
3. Go to `Settings>Labeling Interface`
4. Paste the contents of `label-studio.xml`
5. Head back to the project
6. Import `posts.json`
7. Have fun annotating
8. Export

*If you want to use your manual data for fine tuning, you need to load it in `train_model.py` into the `fine_loader`.
As its format is different from the echr data, you also need to add `hint='reddit'` to the arguments of `get_loader`.*

#### Load the environment for Deepseek
```
ml load GCCcore/.13.3.0
ml load Python/3.12.3
```
*I chose a quite large deepseek distillation, so I could only run this on the cluster and have thus no `nix` environment.*

#### Create the virtual environment for Deepseek
```
python -m venv venv/deepseek
source venv/deepseek/bin/activate
pip install -r requirements/deepseek.txt
deactivate
```

#### Run the Deepseek
```
sbatch deepseek-slurm.sh
```
*Make sure to adapt the time and memory requirements of the slurm Job to fit your needs.*

*The script runs `runner.py`. In there are three sections, one for **annotation** one for **verification** and one for **classification**.
The annotation phase finds relevant entities which are then verified in the verification phase and finally classified as confidential or the like in the classification phase.
Every phase depends on the output of the previous one.
So make sure to (un)comment the right code and run the phases in order.*

#### Convert the Deepseek
```
source venv/deepseek/bin/activate # This script is so simple, it should run in any of the environments
python -m src.converter
deactivate
```
*You will need this to work with the deepseek output in the TAB, as its format is again different from echr and label studio.
This script basically converts it to the echr format.*

#### Aggregate the data
```
source venv/deepseek/bin/activate # This script is so simple, it should run in any of the environments
python -m src.aggregate
deactivate
```
*Technically, you don't need this.
But until now everything is a `.json`.
This script converts the various outputs into two `.csv` files.
One for the sentiment analysis and one for the privacy leakage.*

## Reproducing
I try to be as reproducible as possible, by conducting my experiments in an isolated `nix develop` and a python `venv`. I also try to maintain an up to date copy of `requirements.txt`.
