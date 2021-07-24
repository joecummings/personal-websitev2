< [All tutorials](./tutorials_index.html)

# How to fine-tune BERT using HuggingFace

Joe Cummings

21 July, 2021 (Last Updated: 21 July, 2021)
 
***

* [Motivation](#motivation)
* [Background](#background)
* [Fine-tuning BERT](#fine-tune)
  * [Setup](#setup)
  * [Preprocessing](#preprocessing)
  * [Training & Evaluation](#training-eval)
* [Conclusion](#conclusion)

***

### **Motivation** {#motivation}
BERT is built upon a machine learning architecture called a Transformer and Transformers are sick and tight. Also, everyone from those just flirting with NLP to those on the cutting edge will have to use a Transformer-based model at some point in their lives.

> I'd highly recommend reading through this entire post as it adds color to the model you'll be building, but if you just want the TL;DR, you can skip to the [tutorial part](#fine-tune) now.

### **Background** {#background}
The Transformer architecture was introduced in the paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) in 2017 and has since been cited over 24k times. The Transformer has proven to be both superior in quality and faster to train by virtue of relying solely on [attention mechanisms](https://en.wikipedia.org/wiki/Attention_(machine_learning)) - doing away with cumbersome convolution and recurrence. I'd highly recommend reading more about it before going further with BERT - see these amazing resources [here](https://nlp.seas.harvard.edu/2018/04/03/attention.html) and [here](https://jalammar.github.io/illustrated-transformer/).

> Extra challenge: please [email me](mailto:jrcummings27@gmail.com) if you have an intuitive way to explain **positional encoding** because it still trips me up sometimes.

Researchers jumped at the chance to build upon the Transformer architecture and soon the world had The Allen Institute's [ELMo](https://allennlp.org/elmo) and OpenAI's [GPT/GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf). For this tutorial, we'll look at Google's [BERT](https://arxiv.org/abs/1810.04805), which was introduced in 2019. 

So, what sets BERT apart? The answer lies in the name: **B**idirectional **E**ncoder **R**epresentations from **T**ransformers. Previous models, like GPT, encoded sequences using a left-to-right architecture, meaning that every token can only "see" the tokens that came before it. This is sub-optimal for tasks like question answering where it is extremely helpful to integrate context from the entire sequence. BERT's architecture enables deep bidirectional representations. 

![*Comparison of self-attention - what BERT uses - and masked attention - what GPT uses ([source](https://jalammar.github.io/illustrated-gpt2/))*](https://jc-tutorials.s3.us-east-2.amazonaws.com/fine-tune-bert/self-attention-and-masked-self-attention.png)

You may be thinking at this point: "Okay, why mask the attention mechanism? Why not just integrate the context from the entire sequence from the start?". The answer is that strictly bidirectional conditioning would allow each token in a sequence to essentially "see itself" and the output would be trivially predicted. Imagine I gave you the following sentence: "The dog went to the park" and asked you to "predict" what word came after "dog". Since you have the entire sentence as context, you *know* that "went" immediately succeeds "dog". While this is a slight oversimplification, it should convey the general idea. The diagram below also helps visualize the difference between these langauge modeling methods.

![*Encoding styles of BERT, GPT, and ELMo. ELMo does a shallow concatenation of a left-to-right encoding and a right-to-left encoding.([source](https://arxiv.org/abs/1810.04805))*](https://jc-tutorials.s3.us-east-2.amazonaws.com/fine-tune-bert/model-comparisons.png)

So, if bidirectional encoding is impossible, how is BERT doing it? BERT introduces something called a **"masked language model"** (MLM), but you might also see this referrred to as a [cloze](https://aclanthology.org/W10-1007.pdf) task. In pre-training, 15% of all tokens are replaced with a special `[MASK]` token or a random token. 

```
The dog went to the park. -> The dog [MASK] to the park.
                          -> The dog banana to the park.
```
*Example of how the sequence "The dog went to the park" would be masked in pre-training of BERT.*

The model then is tasked with predicting the correct missing token. So rather than processing the left context of a sequence and trying predict the next token, BERT has to learn how to predict at random spots in the sentence. 

While MLM models the relationship between tokens in a sequence, BERT is also trained on with something called **"next sentence prediction"**, which models the relationships between sentences. This is very useful for question answering, summarization, and multiple-choice tasks. The data is encoded as shown below. 

```
A: The dog went to the park. 
B: It rolled around in the grass.
Classification: IsNext
---
A. The dog went to the park.
B: The crow caws at midnight.
Classification: NotNext
```

These two tasks were trained with 800M words from the [BooksCorpus](https://arxiv.org/abs/1805.10956) and the entirety of English Wikipedia, made up of over 2,500M words. Together, these make up the amazing model that is BERT.

Enough background - let's get to using BERT!

### **Fine-tuning BERT** {#fine-tune}

There are many different ways in which we could load the BERT model and fine-tune it on a downstream task. Some excellent tutorials with different frameworks can be found below:

* Tensorflow: [Fine-tuning a BERT model](https://www.tensorflow.org/official_models/fine_tuning_bert)
* PyTorch: [BERT Fine-Tuning Tutorial with PyTorch](https://mccormickml.com/2019/07/22/BERT-fine-tuning/)

For this tutorial, we'll be using the popular Transformers library from [HuggingFace](https://huggingface.co) to fine-tune BERT on a sentiment analysis task. Despite the slightly silly name, HuggingFace is a fantastic resource for those in NLP engineering. To start, let's create a [conda environment](https://docs.conda.io/en/latest/) and install the HuggingFace library. To support the HuggingFace library, you'll also need to download PyTorch.

#### **1. Setup** {#setup}

```{.bash}
conda create env --name fine-tune-bert python=3.7
conda activate fine-tune-bert
pip install transformers
pip install torch
```

In addition, we'll need to download HuggingFace's Datasets package, which offers easy access to many benchmark datasets. Later on, we're also going to want to specify our own evaluation metric and for that, we need to use [scikit-learn](https://scikit-learn.org/stable/)'s library, so go ahead and install that, too.

```{.bash}
pip install datasets
pip install sklearn
```

Now that we've got everything installed, let's start the actual work. First, let's import the necessary functions and objects.

```{.python}
from datasets import load_dataset, load_metric
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    Trainer,
    TrainingArguments,
)
```

You'll notice that we are importing a version of BERT called `BertForSequenceClassification`. HuggingFace offers several versions of the BERT model including a base `BertModel`, `BertLMHeadMoel`, `BertForPretraining`, `BertForMaskedLM`, `BertForNextSentencePrediction`, `BertForMultipleChoice`, `BertForTokenClassification`, `BertForQuestionAnswering`, and more. The only real difference between a lot of these is the extra layer on top of the pretrained model which is task-specific. You can find all of those models and their specifications [here](https://huggingface.co/transformers/model_doc/bert.html). We're using `BertForSequenceClassification` because we are trying to classify a sequence of text with a certain emotion/sentiment.

The dataset we're using is called `"emotion"` on HuggingFace's Datasets catalog and consists of 20k Tweets labeled for one of 8 emotions: anger, anticipation, disgust, fear, joy, sadness, surprise, and trust. You can read more details about how the data was collected, different baseline experiments, and the data distribution from the [paper](https://aclanthology.org/D18-1404/). So let's load in the `emotions` dataset. 

```{.python}
emo_dataset = load_dataset("emotion")  # It really is that easy.
```

Take a peek at the first 5 items in the training data and see what we have.

```{.python}
>>> emo_dataset["train"][:5]
{
    'text': [
        'i didnt feel humiliated',
        'i can go from feeling so hopeless to so damned hopeful just from being around someone who cares and is awake',
        'im grabbing a minute to post i feel greedy wrong',
        'i am ever feeling nostalgic about the fireplace i will know that it is still on the property',
        'i am feeling grouchy'
    ], 
    'label': [0, 0, 3, 2, 3]
}
```

It appears the text has already been lowercased (good!), links and hastags have been removed, and contractions are standardized. It's also good to double check that the data makes sense. The labels have already been converted into a numeric value, with each number corresponding to an emotion. For example, `0` is "sadness", `3` is "anger", and `2` is "love".

#### **2. Preprocessing** {#preprocessing}

Now that we have some data, we need to do some preprocessing to it so that BERT can understand and thankfully, HuggingFace provides a helpful `BertTokenizer` that takes care of this for us.

We can load the BERT Tokenizer from a pretrained model (they come together).
```{.python}
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
```

And let's see how this is encoded!
```{.python}
>>> tokenizer(["The dog went to the park."], padding="max_length", truncate=True)
{
    'input_ids': [[101, 1996, 3899, 2253, 2000, 1996, 2380, 1012, 102, 0, ..., 0]], 
    'token_type_ids': [[0, ..., 0]], 
    'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, ..., 0]]
}
```

At this point, you might be thinking what the hell am I looking at? Well I've spared your eyes by truncating the number of zeros shown in the output, but the length of each item in the returned dictionary is 512, which is the maximum length that BERT can accept. To some, this might be confusing - why do we need every input sequence to be the same size? The answer is efficiency. Linear combinations are much faster than normal multiplication and for those to be possible, all vectors need to be of the same length. 

* `input_ids` correspond to a given token in the vocabulary. BERT also contains a set of tokens to denote special meanings.
  * `[CLS]`: Short for classification. Goes at the beginning of every sequence.
  * `[SEP]`: Short for separation. Goes in between sentences if given a sentence pair, and at the end of the sequence.
  * `[UNK]`: Short for unknown. Replaces any tokens that cannot be found in the vocabulary.

So the mapping from above looks like the following:

token input_id
----- ---------
[CLS] 101
the   1996
dog   3899
went  4
to    2000
park  2380
.     1012
[SEP] 102
----- ---------

* `token_type_ids` are used in any tasks that contain two sequences, like question answering, summarization, etc. Because we are doing a sequence classification task with only one sentence, all our `token_type_ids` will be `0`.
* `attention_mask` refers to which tokens the model should "attend". For the most basic case, we want the model to be able to "see" all of our tokens, which are marked with a `1`. 

Keep in mind while debugging, that you may see more `input_ids` than original tokens. That's because BERT tokenizes using [WordPiece](https://arxiv.org/abs/1609.08144v2) which can split some words into two or three different tokens.

So, how can we apply this tokenize function across all text labels? One way would simply be to iterate programmatically over every entry in the dataset and convert the text like so:
```{.python}
for example in emo_dataset["train"]:
    tokenized_text = tokenizer(example["text"])
```
This is a slow process and we do have a better option. HuggingFace provides us with a useful [`map`](https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map) function on the `Dataset` object. 

```{.python}
def tokenize_go_emotion(example):
    return tokenizer(example["text"], padding="max_length", truncation=True)

tokenized_data = dataset.map(tokenize_go_emotion, batched=True)
```

In my experience, `Dataset.map` runs ~200ms faster than linear iteration and automatically caches the result so that each subsequent call takes a fraction of the time to complete.

> Bonus question: why wouldn't we want to just define our function as `lambda x: tokenizer(x["text"])` and save a couple lines of code? (See answer at the end of the tutorial).

As mentioned, `emotion` is a rather large dataset and I don't know about you, but I'm trying not to beat the shit out of my already overworked computer. So let's shuffle the data and grab a subset of the examples.

```{.python}
small_train_dataset = tokenized_data["train"].shuffle(seed=42).select(range(1000))
small_val_dataset = tokenized_data["validation"].shuffle(seed=42).select(range(100))
```

Now for the fun part - let's build this model! First, we load BERT from a pretrained HuggingFace location and specify how many labels BERT will have.

```{.python}
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=8
)
```

#### **3. Training & Evaluating** {#training-eval}

Second, we load the training arguments for the model (this could also be known as the config). `TrainingArguments` has a ton of parameters so you can check those out [here](https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments). For our purposes, we only need to specify the output directory, the evaluation strategy (when will we evaluate the results), and the number of epochs to run.

```{.python}
training_args = TrainingArguments(
    "bert_trainer",
    evaluation_strategy="epoch",
    num_train_epochs=5,
)
```

Finally, we're ready to train. We set up an abstract Trainer class and give it our model, arguments, the training dataset, and the validation dataset to evaluate on. Calling the `trainer.train()` method (not surprisingly) kicks of the model fine-tuning.

```{.python}
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_val_dataset,
)

trainer.train()
```

The first output will probably look something like...

```
Some weights of the model checkpoint at bert-base-uncased were not used when initializing BertForSequenceClassification: ['cls.predictions.transform.LayerNorm.bias', 'cls.predictions.transform.dense.bias', 'cls.seq_relationship.weight', 'cls.predictions.decoder.weight', 'cls.seq_relationship.bias', 'cls.predictions.bias', 'cls.predictions.transform.dense.weight', 'cls.predictions.transform.LayerNorm.weight']
- This IS expected if you are initializing BertForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing BertForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
Some weights of BertForSequenceClassification were not initialized from the model checkpoint at bert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
```

But don't freak out! This is what we expect because we are randomly initializing the weights of the last head for our classification task.

If you have access to a GPU, HuggingFace will automatically find the device and push most calculations to that. I was able to run the entire dataset on a single Nvidia GeForce GTX 1080 GPU in 77 minutes with an evalutation `micro f1` score of `94%`. 

> If you're unfamiliar with the F1-scoring metric, you can read more about it [here](https://machinelearningmastery.com/classification-accuracy-is-not-enough-more-performance-measures-you-can-use/) and why it can be a better metric than accuracy.

I recognize that not all people have access to such compute power, so for comparison, I ran the fine-tuning on an `Intel(R) Xeon(R) CPU E5-2630 v4 @ 2.20GHz`. Unfortunately, the amount of time this would take is somewhat absurd, so I scaled back the size of our dataset. Training on 1000 examples, I fine-tuned BERT in 1020 minutes with an evaluation `micro f1` score of 86%.

### Conclusion {#conclusion}
In this tutorial, we learned about the incredible Transformer model called BERT and how to quickly and easily fine-tune it on a downstream task. With this knowledge, you can go forth and build many a NLP application.

> Bonus question answer: the [`pickle`](https://docs.python.org/3/library/pickle.html) module, which is the default serializer in Python, does not serialize or deserialize code, e.g. lambda functions. It only serializes the names of classes/methods/functions. Therefore, if you want to save your model to use again, you cannot use an anonymous function.

You can find all the code for this tutorial on my [Github](https://github.com/joecummings/fine-tune-bert
). If you have any comments, questions, or corrections, feel free to [drop me a line](mailto:jrcummings27@gmail.com).

#### Thanks to:

[Dan Knight](https://www.linkedin.com/in/dan-c-knight/) for his feedback and encouragement.
