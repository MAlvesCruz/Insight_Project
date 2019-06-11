# Find My Expert
### The best fit for your needs

Consulting project with a former Brazilian Q&A website.

## Problem

Nowadays one can find many w&A websites in order to get the information needed. However, many of these services are provided mostly (or only) in English. For instance, support for Portuguese only became available last year for **quora**, even though it is the seventh most spoken language in the world. 

![title](./images/linguas_esta.jpg)


Furthermore, the idea behind a Q&A website is that anyone can, in principle, reply to any question. 
Back in 2011, a Brazilian Q&A website was launched with the purpose of finding people with the "right" knowledge to provide the best answers to those seeking certain information. 

Because at the time there was no (or limited) association of tags and keywords to a question and/or user, the recommendation system of the website presented some issues and only half of the questions posted were ever answered.

## Solution

In order to make a better recommendation system, Natural Language Processing is going to be applied to the archive data. The archive data contains information regarding all the questions that were ever asked in the website:

1) The question body
2) The time it was created
3) The time it was answered
4) Which users answered the question

### Challenges

All the tools available for NLP work, in general, pretty well for English texts. However, for non-English languages they are limited or lacking. One example of this is Lemmatization, which is the assignment of a conjugated verbe to its infinitive form, is lacking for Portuguese.  This poses a problem, specially because Portuguese, as many Latin languages, has a lot of inflections. The Figure below shows the verb to be conjugated in the **mode Indicative** in Portuguese.

![title](./images/conjuagacao_ser.jpg)
