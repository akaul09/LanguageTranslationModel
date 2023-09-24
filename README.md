Utilizing a BERT model to create a Language Translation model for French and English.

## Contents
SRC folder - containing source code for our EDA and model.

FIGURES folder - containing figures/visualizations of our data.

LICENSE.md - MIT License.

README.md - The current file.

More details below.

## SRC
DS 4002 Dataset/EDA - Code files containing our initial EDA and visualizations.
Project1.ipynb - Jupyter Notebook containing our BERT model analysis


### Installing/Building the model
Make sure you have the latest version of python installed, with a IDE capable of opening jupyter notebooks. VSCode can work. Make sure all necessary libraries are installed as well. Most should be automatically installed once our code is run. From there, copy the dataset from our drive link to the same location as Project1.ipynb. 

### Usage of the code
Each code block can be run in sequential order, and the model should work. Training time may vary depending on dataset size and number of epochs.

## DATA
Data (eng_-french.csv) is stored in the DATA folder.

| **Column**       | **Type**     | **Description** |
|--------------|-----------|------------|
| English words/sentences | string  | An English word or sentence        |
| French words/sentences | string  | The French translation of the English word or sentence       |

Data was split into a training set and testing set for a total of 35000 rows. The training set contained 25000 rows and was used to train the model. The testing set contained 10000 rows and was used to test the results of the model.

## FIGURES
| **Figure**       | **Description**     | **Takeaways** |
|--------------|-----------|------------|
| String Length Frequency by Language | Bar chart displaying the frequencies of string length in phrases grouped by language | There are significantly more shorter and mid-sized English phrases than French phrases and there are significantly more longer French phrases than English phrases. There are no English phrases in our dataset with 25 or more characters, but there are a notable amount of French phrases.   |
| Distribution of String Length by Language | Boxplots displaying the distribution of string length by language  |   French phrases have a slightly higher median length (at around 21 characters) than English phrases (at around 17 characters). The French phrases also seem to have a slightly wider distribution in string length (i.e. longer phrases) than English phrases, as its interquartile range seems to be larger and its maximum length is also higher. |

## REFERENCES
[1]	“What are the Romance Languages? | Romance Languages,” www.rom.uga.edu. https://www.rom.uga.edu/what-are-romance-languages

[2]	N. Donges, “What is transfer learning? Exploring the popular deep learning approach,” Built In, Aug. 25, 2022. https://builtin.com/data-science/transfer-learning

‌[3]	“Evaluating models | AutoML Translation Documentation,” Google Cloud. https://cloud.google.com/translate/automl/docs/evaluate

[4]	“English-French Translation Dataset,” www.kaggle.com. https://www.kaggle.com/datasets/dhruvildave/en-fr-translation-dataset (accessed Sep. 
09, 2023).

[5]	“Language Translation (English-French),” www.kaggle.com. https://www.kaggle.com/datasets/devicharith/language-translation-englishfrench (accessed Sep. 09, 2023).

‌[6]	Rani Horev, “BERT Explained: State of the art language model for NLP,” Medium, Nov. 10, 2018. https://towardsdatascience.com/bert-explained-state-of-the-art-language-model-for-nlp-f8b21a9b6270

‌[7]	“Global language services: market size 2019 | Statista,” Statista, 2019. Https://www.statista.com/statistics/257656/size-of-the-global-language-services-market/

[8]	B. Zoph, D. Yuret, J. May, and K. Knight, “Transfer Learning for Low-Resource Neural Machine Translation,” Association for Computational Linguistics, 2016. Accessed: Sep. 09, 2023. [Online]. Available: https://aclanthology.org/D16-1163.pdf

[9]	Y. Wan et al., “Challenges of Neural Machine Translation for Short Texts,” Computational Linguistics, pp. 1–21, Mar. 2022, doi: https://doi.org/10.1162/coli_a_00435.

[10] 	“BERT Transformers for Natural Language Processing,” Paperspace Blog, May 18, 2022. https://blog.paperspace.com/bert-natural-language-processing/#:~:text=Neural%20Machine%20Translation%20-%20The%20BERT (accessed Sep. 24, 2023).

[11] 	S. A. G. Shakhadri, “Language Translation with Transformer In Python!,” Analytics Vidhya, Jun. 12, 2021. https://www.analyticsvidhya.com/blog/2021/06/language-translation-with-transformer-in-python/ (accessed Sep. 24, 2023).
 
[12] 	“bert-base-multilingual-cased · Hugging Face,” huggingface.co. https://huggingface.co/bert-base-multilingual-cased
 
‌‌[13]	“BERT Fine-Tuning Tutorial with PyTorch · Chris McCormick,” Mccormickml.com, Jul. 22, 2019. https://mccormickml.com/2019/07/22/BERT-fine-tuning/


### Acknowledgements
Professor Alonzi

Harsh (TA)

Group12

### Previous Works
MI1: https://docs.google.com/document/d/1-xsHooWhw5ovA3JCjLr3xrNjGehp26iIPEn9VmnKDuc/edit

MI2: https://docs.google.com/document/d/1GIFuxAvRrH3NDUuGEGP8eFPUY5R6jeHDxtFH1KPfRNQ/edit 

This project is licensed under the terms of the MIT license.
