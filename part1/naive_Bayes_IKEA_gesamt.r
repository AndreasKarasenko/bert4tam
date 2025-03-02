# Naive Bayes mit den IKEA gesamt Daten
# in Anlehnung an https://rpubs.com/meisenbach/272229 

# List of packages
packages <- c("tidyverse", "readr", "RColorBrewer", "tidytext", "wordcloud", 
              "dplyr", "lsa", "reshape2", "readxl", "tm", "e1071", "gmodels", "reshape2")

# Function to check and install packages
check_and_install <- function(pkg){
  if (!require(pkg, character.only = TRUE)){
    install.packages(pkg, dependencies = TRUE)
  }
}

# Apply the function to each package
sapply(packages, check_and_install)
library(tidyverse); library(readr); library(RColorBrewer)
library(tidytext); library(wordcloud); library(dplyr); library(lsa)
library(reshape2); library(readxl); library(tm); library(e1071)
library(gmodels); library(reshape2)
getwd()

# Step 1: Data
# setwd("C:/Users/Alpha/Documents/Papers/__ICIS_2023/Baier_Karasenko_Rese/Data")
Daten=read_excel("ikea1.xlsx",na="NA")
j.content=2; j.score=6; j.score2=j.score+7; j.score3=j.score2+7;
Daten=Daten[,c(j.content,j.score,j.score2,j.score3)]; head(Daten)
Daten2=Daten; Daten2[,2]=Daten2[,3]
Daten3=Daten; Daten3[,2]=Daten3[,3]
Daten=rbind(Daten,Daten2,Daten3)
Daten=Daten[,c(1,2)]; dim(Daten)
names(Daten)=c("Text","score"); head(Daten)

Daten$Wertung="Neutral" # Kommentare mit Wertung >=4
Daten$Wertung[Daten$score<3]="Negativ" # 1 oder 2
Daten$Wertung[Daten$score>3]="Positiv" # 4 oder 5
I=dim(Daten)[1]; Daten$scoreb="3"
for (i in 1:I){Daten$scoreb[i]=as.character(round(Daten$score[i]))}
Daten$scoreb=as.factor(Daten$scoreb)
head(Daten); str(Daten); Daten=as_tibble(Daten)

tidy_Daten <- Daten %>%
  select(Text,Wertung) %>%
  unnest_tokens("word",Text) # Alle Wörter in einer Spalte
tidy_Daten1 <- tidy_Daten %>%
  mutate(word_stem=SnowballC::wordStem(word,language="german"))
tidy_Daten1[1:10,]

# Auszählen der Wörter
tidy_Daten1 %>%
  count(word) %>%
  arrange(desc(n))

# Stopwörter entfernen mit Paket lsa
data(stopwords_de); stopwords_de<-data_frame(word=stopwords_de)
tidy_Daten <- Daten %>%
  select(Text,Wertung) %>%
  unnest_tokens("word",Text) # Alle Wörter in einer Spalte
tidy_Daten %>% 
  anti_join(stopwords_de) -> tidy_Daten # Stoppwörter entfernen
tidy_Daten1 <- tidy_Daten %>%
  mutate(word_stem=SnowballC::wordStem(word,language="german"))
tidy_Daten1 %>%
  count(word) %>%
  arrange(desc(n))

# Wordcloud
# dev.off() # evtl. Graphikparameter zurücksetzen
tidy_Daten1 %>%
  count(word) %>%
  with(wordcloud(word, n, max.words = 200))

# Comparison Cloud
dev.off() # Graphikparameter zurücksetzen
posneg=as.data.frame(tidy_Daten1); dim(posneg)
posneg=posneg[posneg$Wertung=="Positiv" | posneg$Wertung=="Negativ",]; dim(posneg)
posneg = as_tibble(posneg) %>%
  count(word, Wertung, sort = TRUE) %>%
  acast(word ~ Wertung, value.var = "n", fill = 0) 
posneg=as.data.frame(posneg)
pos=posneg[order(-posneg$Positiv),]; pos[1:200,]
neg=posneg[order(-posneg$Negativ),]; neg[1:200,]
posneg[,c(1,2)] %>%
  comparison.cloud(colors = c("red", "black"),
                   max.words = 200)

### Naive Bayes

# Create random samples
set.seed(123)
train_index <- sample(I,round(0.8*I))
Daten.train <- Daten[train_index, ]
Daten.test  <- Daten[-train_index, ]
prop.table(table(Daten.train$scoreb))

# Create Corpi for train and test
corpus.train <- VCorpus(VectorSource(Daten.train$Text),
                        readerControl = list(language = "de"))
corpus.test <- VCorpus(VectorSource(Daten.test$Text),
                       readerControl = list(language = "de"))

# Document-Term-Matrix for train and test
dtm.train <- DocumentTermMatrix(corpus.train, control = list(
  tolower = TRUE,
  removeNumbers = TRUE,
  stopwords = TRUE,
  removePunctuation = TRUE,
  stemming = TRUE
))
dtm.test <- DocumentTermMatrix(corpus.test, control = list(
  tolower = TRUE,
  removeNumbers = TRUE,
  stopwords = TRUE,
  removePunctuation = TRUE,
  stemming = TRUE
))

# create function to convert counts to a factor
convert_counts <- function(x) {
  x <- ifelse(x > 0, "Yes", "No")
}
# apply() convert_counts() to columns of data
dtm_binary.train <- apply(dtm.train, MARGIN = 2, convert_counts)
dtm_binary.test <- apply(dtm.test, MARGIN = 2, convert_counts)

# Step 3: Training a model on the data
classifier <- naiveBayes(as.matrix(dtm_binary.train), Daten.train$scoreb)

# Step 4: Evaluating model performance
classifier_pred.train <- predict(classifier, as.matrix(dtm_binary.train))
head(classifier_pred.train)
CrossTable(classifier_pred.train, Daten.train$scoreb,
           prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,
           dnn = c('predicted', 'actual'))
classifier_pred.test <- predict(classifier, as.matrix(dtm_binary.test))
cor(as.numeric(classifier_pred.train),Daten.train$score)
CrossTable(classifier_pred.test, Daten.test$scoreb,
           prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,
           dnn = c('predicted', 'actual'))
cor(as.numeric(classifier_pred.test),Daten.test$score)

classifier_pred.train.raw <- predict(classifier, as.matrix(dtm_binary.train),type="raw")
classifier_pred.test.raw <-  predict(classifier, as.matrix(dtm_binary.test),type="raw")
I.train=dim(classifier_pred.train.raw)[1]
I.test=dim(classifier_pred.test.raw)[1]
classifier_pred.train.wmean=classifier_pred.train.raw[,1]
classifier_pred.test.wmean=classifier_pred.test.raw[,1]
for (i in 1:I.train){
  zsum=0; nsum=0
  for (j in 1:5){
    zsum=zsum+j*classifier_pred.train.raw[i,j]
    nsum=nsum+classifier_pred.train.raw[i,j]
  }
  classifier_pred.train.wmean[i]=zsum/nsum
}
cor(classifier_pred.train.wmean,Daten.train$score)
for (i in 1:I.test){
  zsum=0; nsum=0
  for (j in 1:5){
    zsum=zsum+j*classifier_pred.test.raw[i,j]
    nsum=nsum+classifier_pred.test.raw[i,j]
  }
  classifier_pred.test.wmean[i]=zsum/nsum
}
cor(classifier_pred.test.wmean,Daten.test$score)

if (!require(caret, character.only = TRUE)){
  install.packages("caret", dependencies = TRUE)
}
library(caret)# Calculate the confusion matrix
cm <- confusionMatrix(as.factor(classifier_pred.test), as.factor(Daten.test$score))

# Print the overall statistics
print(cm$overall)

# Print the class statistics
print(cm$byClass)