# multilabel-classification-nlp-r

## 📑Description
Hi there! 

This project is dedicated to solving the problem of automating content analysis in employee engagement research. The employee engagement survey in this project was carried out using the method of open questions. 16,000 employees of five Russian companies were asked two open-ended questions indirectly ascertaining their engagement:
1) What do you think makes our Company a good employer?
2) What do you think should be improved in our Company to make it a better employer?

The categories used for classification were obtained by building a co-occurrence matrix from the results of manual content analysis. 
The tool consists of two independent models based on **supervised multilabel classification** and **NLP**. 
The project is written in R Language.


## 📁 Files
The files below contain two sheets ("Good employer" and "Development zones") for each of the two open questions, respectively.

- **"bigdata.xlsx"** (Under NDA) — Each sheet contains two columns: the respondent's ID and the text of his answer.
- **"categoties.xslx"** — Each sheet contains a table with the respondent's ID and several columns with categories identified by expert coders who manually coded the texts of the answers. 0 or Null indicates the absence of a category in the respondent's answer, and 1 indicates the presence of a category.

## 💡Findings
The solution contains two separate models developed with supervised machine learning algorithms – random forest and gradient boosting. These models automate the content analysis process for answers to open questions in engagement studies with the accuracy of .86 and .70 respectively. Aggregated and potentially universal categories of engagement studies were identified, the categories being independent of the context of data collection.  

Besides, the most important speech patterns (words and phrases) that predict to which of the categories a particular employee's response would be assigned were named. The solution can be used by companies for benchmarking factors of engagement in different domain areas.

## ✅ Status
Completed!
