install.packages("BiocManager") 
BiocManager::install("WGCNA")
install.packages(c("flashClust"))

library(WGCNA)
library(flashClust)

# Load training dataset
dfTraining = read.csv('../data/nci_almanac_preprocessed/omics/split/merged_rnaseq_fpkm_prot_coding_train.csv.gz')
dfTraining = dfTraining[, !(colnames(dfTraining) %in% c("CELLNAME"))]

# # # WGCNA
# # The following code was copied from this tutorial: https://horvath.genetics.ucla.edu/html/CoexpressionNetwork/Rpackages/WGCNA/Tutorials/FemaleLiver-02-networkConstr-auto.pdf
# # Choose a set of soft-thresholding powers
# powers = c(c(1:10), seq(from = 12, to=20, by=2))
# # Call the network topology analysis function
# sft = pickSoftThreshold(dfTraining, powerVector = powers, verbose = 5)
# # Plot the results:
# pdf(file = "../soft_threshold.pdf",   # The directory you want to save the file in
#     width = 4, # The width of the plot in inches
#     height = 4) # The height of the plot in inches
# sizeGrWindow(9, 5)
# par(mfrow = c(1,2));
# cex1 = 0.9;
# # Scale-free topology fit index as a function of the soft-thresholding power
# plot(sft$fitIndices[,1], -sign(sft$fitIndices[,3])*sft$fitIndices[,2],
#      xlab="Soft Threshold (power)",ylab="Scale Free Topology Model Fit,signed R^2",type="n",
#      main = paste("Scale independence"));
# text(sft$fitIndices[,1], -sign(sft$fitIndices[,3])*sft$fitIndices[,2],
#      labels=powers,cex=cex1,col="red");
# # this line corresponds to using an R^2 cut-off of h
# abline(h=0.90,col="red")
# # Mean connectivity as a function of the soft-thresholding power
# plot(sft$fitIndices[,1], sft$fitIndices[,5],
#      xlab="Soft Threshold (power)",ylab="Mean Connectivity", type="n",
#      main = paste("Mean connectivity"))
# text(sft$fitIndices[,1], sft$fitIndices[,5], labels=powers, cex=cex1,col="red")
# dev.off()

# # the following code is based on this tutorial: https://horvath.genetics.ucla.edu/html/CoexpressionNetwork/JMiller/
softPower = 14
adjacency_train = adjacency(dfTraining, power=softPower,type="signed");
diag(adjacency_train)=0
dissTOM_train = 1-TOMsimilarity(adjacency_train, TOMType="signed")
geneTree_train = flashClust(as.dist(dissTOM_train), method="average")
# tree = cutreeHybrid(dendro = geneTree_train, pamStage=FALSE,
#                     minClusterSize = (30-3*0), cutHeight = 0.99,
#                     deepSplit=0, distM = dissTOM_train)
tree = cutreeHybrid(dendro = geneTree_train, pamStage=FALSE,
                    minClusterSize = 20, cutHeight = 0.99,
                    deepSplit=3, distM = dissTOM_train) # deepsplit can be 0-4

modules_train = labels2colors(tree$labels)
PCs_train = moduleEigengenes(dfTraining, colors=modules_train, excludeGrey=TRUE)
rm(dfTraining)
ME_train = PCs_train$eigengenes
write.csv(ME_train, file='../data/nci_almanac_preprocessed/omics/train_module_eigengenes.csv', row.names = FALSE)

# Load validation dataset and calculate moduleEigengenes for modules identified in the training set
dfValidation = read.csv('../data/nci_almanac_preprocessed/omics/split/merged_rnaseq_fpkm_prot_coding_val.csv.gz')
dfValidation = dfValidation[, !(colnames(dfValidation) %in% c("CELLNAME"))]
PCs_val = moduleEigengenes(dfValidation, colors=modules_train, excludeGrey=TRUE)
rm(dfValidation)
ME_val = PCs_val$eigengenes
write.csv(ME_val, file='../data/nci_almanac_preprocessed/omics/val_module_eigengenes.csv', row.names = FALSE)

# Load test dataset and calculate moduleEigengenes for modules identified in the training set
dfTest = read.csv('../data/nci_almanac_preprocessed/omics/split/merged_rnaseq_fpkm_prot_coding_test.csv.gz')
dfTest = dfTest[, !(colnames(dfTest) %in% c("CELLNAME"))]
PCs_test = moduleEigengenes(dfTest, colors=modules_train, excludeGrey=TRUE)
rm(dfTest)
ME_test = PCs_test$eigengenes
write.csv(ME_test, file='../data/nci_almanac_preprocessed/omics/split/test_module_eigengenes.csv', row.names = FALSE)

# Load full dataset and calculate moduleEigengenes for modules identified in the training set
dfFull = read.csv('../data/nci_almanac_preprocessed/omics/merged/merged_rnaseq_fpkm_prot_coding.csv.gz')
dfFull = dfFull[, !(colnames(dfFull) %in% c("CELLNAME"))]
PCs_full = moduleEigengenes(dfFull, colors=modules_train, excludeGrey=TRUE)
#PCs_full = moduleEigengenes(dfFull, colors=modules_train, trapErrors=TRUE, returnValidOnly=TRUE)
rm(dfFull)
ME_full = PCs_full$eigengenes
write.csv(ME_full, file='../data/nci_almanac_preprocessed/omics/full_module_eigengenes.csv', row.names = FALSE)
