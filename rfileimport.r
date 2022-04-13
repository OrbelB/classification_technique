library(keras)
#read matlab
library(R.matlab)
#open cv port for r
library("opencv")
library(filesstrings)
install.packages("filesstrings")
library(stringr)
#mona <- ocv_read('https://jeroen.github.io/images/monalisa.jpg')
#mona <- ocv_resize(mona, width = 320, height = 477)
# FAST-9
#pts <- ocv_keypoints(mona, method = "FAST", type = "TYPE_9_16", threshold = 40)
# Harris
#pts<- ocv_keypoints(mona, method = "Harris", maxCorners = 128)
# Convex Hull of points
#pts <- ocv_chull(pts)

#pts <- ocv_hog(ocv_grayscale(mona))


#read matlab matrix
train_list_from_mat = readMat('/Users/marvinharootoonyan/Desktop/Comp541/lists/train_list.mat')
train_list_from_mat$labels


w <- c()
#for loop of file list
for(it in train_list_from_mat$file.list)
{
  l <- paste('/Users/marvinharootoonyan/Desktop/Comp541/Images/',it,sep = "")
  d <- paste('/Users/marvinharootoonyan/Desktop/TrainImages/',it, sep="")
  g <- paste('mkdir -p',paste(paste(" `dirname",d)),"`",sep="")
  w <- append(w,g)
  s <- paste('cp',paste(l,d,sep=" "),sep=" ")
  w <- append(w,s)
}
w

write.table(w, file = "script.txt", sep = "\t",
            row.names = FALSE)
#verify first image
x_train[1]
