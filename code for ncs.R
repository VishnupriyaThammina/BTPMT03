x<-c(1,2,3,4,5,6,7,8,9,10)
library(ggplot2)
z<-nscore(x)

data<-read.csv("C:\\Users\\Santhosh\\Desktop\\premi\\x1.csv")
p<-nscore(data$pm25)
nscore <- function(x) {
  # Takes a vector of values x and calculates their normal scores. Returns 
  # a list with the scores and an ordered table of original values and
  # scores, which is useful as a back-transform table. See backtr().
  nscore <- qqnorm(x, plot.it = FALSE)$x  # normal score 
  trn.table <- data.frame(x=sort(x),nscore=sort(nscore))
  
  return (list(nscore=nscore, trn.table=trn.table))
}

hist(p$nscore,xlab = "Weight",col = "yellow",border = "blue")
