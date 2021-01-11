getwd()
setwd("/Users/inigo/Desktop/Master Data Science/Inteligent systems/NLP")

library(tm)
library(ggplot2)
library(wordcloud)
#library(RWeka)
library(reshape2)
library(textdata)
library(gridExtra)
library(dplyr)

# Text mining part

# We load the data 

path_data = '/Users/inigo/Desktop/Master Data Science/Inteligent systems/NLP'

data <- dataset_ag_news( dir = path_data,
                  split = "train",
                  delete = FALSE,
                  return_path = FALSE,
                  clean = FALSE,
                  manual_download = FALSE
                  )

newsCorpus <- Corpus(VectorSource(data$description)) 
news_DTM <- DocumentTermMatrix(newsCorpus)
inspect(news_DTM) # 100% Sparsity

# We delete the less frequent terms
news_dtm <- removeSparseTerms(news_DTM, 0.99)
inspect(news_dtm) 

# We check words frequency

length(dimnames(news_dtm)$Terms) # How many terms have been identified in the TDM
freq = rowSums(as.matrix(news_dtm))

# We plot words frequency and check the tendency
plot1 <- plot(sort(freq, decreasing = T),col="green",main="Word frequencies (removing sparse term)", xlab="Frequency-based rank", ylab = "Frequency")
findFreqTerms(news_dtm, 100) # The most frequent terms

freq = data.frame(sort(colSums(as.matrix(news_dtm)), decreasing=TRUE)) # The frequent terms sorted
wordcloud(rownames(freq), freq[,1], max.words=100, colors=brewer.pal(5, "Dark2")) # Wordcloud of terms

# We see that the most common words are common but they cant provide a lot of information:
# We apply some transformations:

# Switch to lower case
newsCorpus = tm_map(newsCorpus, content_transformer(tolower))
# Remove numbers
newsCorpus= tm_map(newsCorpus, removeNumbers)
# Remove punctuation marks 
newsCorpus= tm_map(newsCorpus, removePunctuation)
# Remove stopwords
newsCorpus = tm_map(newsCorpus, removeWords, c("the", "and", stopwords("english")))
# Remove extra whitespaces
newsCorpus =  tm_map(newsCorpus, stripWhitespace)

news_DTM <- DocumentTermMatrix(newsCorpus)
inspect(news_DTM) # 100% Sparsity

# We delete the less frequent terms
news_dtm <- removeSparseTerms(news_DTM, 0.99)
inspect(news_dtm)

# We repeat previous plot:

freq = data.frame(sort(colSums(as.matrix(news_dtm)), decreasing=TRUE)) # The frequent terms sorted
wordcloud(rownames(freq), freq[,1], max.words=100, colors=brewer.pal(5, "Dark2")) # Wordcloud of terms
# BETTER RESULTS

# We can plot the most popular words by class:

library(tidytext) # tidy implimentation of NLP methods
library(dplyr)

top_terms_by_class_tfidf <- function(text_df, text_column, group_column, plot = T){
  
  # name for the column we're going to unnest_tokens_ to
  # (We only need to worry about enquo stuff if we're
  # writing a function using using tidyverse packages)
  group_column <- enquo(group_column)
  text_column <- enquo(text_column)
  
  # get the count of each word in each news
  words <- text_df %>%
    unnest_tokens(word, !!text_column) %>%
    mutate_all(as.character) %>% 
    count(!!group_column, word) %>% 
    ungroup()
  
  # get the number of words per text
  total_words <- words %>% 
    group_by(!!group_column) %>% 
    summarize(total = sum(n))
  
  # combine the two dataframes we just made
  words <- left_join(words, total_words)
  
  # get the tf_idf & order the words by degree of relevence
  tf_idf <- words %>%
    bind_tf_idf(word, !!group_column, n) %>%
    select(-total) %>%
    arrange(desc(tf_idf)) %>%
    mutate(word = factor(word, levels = rev(unique(word))))
  
  if(plot == T){
    
    # If Tplot = True we do the plotting
    group_name <- quo_name(group_column)
    
    # plot the 10 most informative terms per topic
    tf_idf %>% 
      group_by(!!group_column) %>% 
      top_n(10) %>% 
      ungroup %>%
      ggplot(aes(word, tf_idf, fill = as.factor(group_name))) +
      geom_col(show.legend = FALSE) +
      labs(x = NULL, y = "tf-idf") +
      facet_wrap(reformulate(group_name), scales = "free") +
      coord_flip()
  }else{
    # return the entire tf_idf dataframe
    return(tf_idf)
  }
}

# We will just use the function to do the plotting:

top_terms_by_class_tfidf(text_df = data, # dataframe
                         text_column = description, # column with text
                         group_column = class, # column with topic label
                         plot = T) # return a plot

#################
#### MODELING ###
#################

# We build several models using the TfIdf matrix 

news_dtm_tfidf <- DocumentTermMatrix(newsCorpus, control = list(weighting = weightTfIdf)) # Create the DMT
news_dtm_tfidf = removeSparseTerms(news_dtm_tfidf, 0.99) # To reduce sparsity
news = data.frame(as.matrix(news_dtm_tfidf)) # Store as dataframe
news$class = as.factor(data$class) # Add the class as new column
news$... <- NULL # Delete a column giving problems
head(news) # To check our new df

# We divide between train/test:
id_train <- sample(nrow(news),nrow(news)*0.80)
news.train = news[id_train,]
news.test = news[-id_train,]

# We try out different models and check their performance:

# Decision trees:
library(rpart)
library(rpart.plot)

news.tree = rpart(class~.,  method = "class", data = news.train)
# Visualize the tree:
prp(news.tree)

prediction <- predict(news.tree, news.test,  type="class")
table(news.test$class,prediction,dnn=c("Observed:","Predicted:"))

# Neural networks
library(nnet)

news.nnet = nnet(class~., data=news.train, size=1, maxit=800)
prediction2 <- predict(news.nnet, news.test,  type="class")
table(news.test$class,prediction2,dnn=c("Observed:","Predicted:"))

# Support vector machine:
library(e1071)

news.svm = svm(class~., data = news.train)
prediction2 <- predict(news.nnet, news.test,  type="class")
table(news.test$class,prediction2,dnn=c("Observed:","Predicted:"))


# k-NN (takes a lot of time):
install.packages("DMwR")
library('DMwR')

news.knn = kNN(class~., news.train,news.test,norm = T) # Normalizing the data
table(news.test$class,news.knn,dnn=c("Observed:","Predicted:"))

