if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")

BiocManager::install("clusterProfiler")
BiocManager::install("org.Hs.eg.db")

library('clusterProfiler')
library(org.Hs.eg.db)
library(enrichplot)
library(stats)

dfShap <- read.csv('..\\results\\shap_analysis\\sample11835_shap_values.csv')
dfShap <- dfShap[,!grepl("drug", colnames(dfShap))]
dfShap["Unnamed..0"] <- NULL
dfShapT = t(dfShap)
rankedgenes <- dfShapT[order(dfShapT, decreasing = T),]
rankedgenes

set.seed(123) 
result <- gseGO(rankedgenes, ont="ALL", OrgDb = org.Hs.eg.db, keyType = "SYMBOL", seed=T, nPermSimple = 5000) # increased number of permutations due to warning about p-value calculation
write.csv(result@result, file="..\\results\\shap_analysis\\rankedgsea_go_sample11835.csv")

#heatplot(result, foldChange=rankedgenes, showCategory=32)
#gseaplot2(result, geneSetID = 1:5)
setEPS()
postscript("..\\results\\shap_analysis\\rankedgsea_sample11835_leukocyte_specific.eps")
gseaplot2(result, geneSetID = c(11, 13, 26, 21)) # leukocyte/lymphocyte-specific gene sets
dev.off()

setEPS()
postscript("..\\results\\shap_analysis\\rankedgsea_sample11835_kinase_gene_sets.eps")
gseaplot2(result, geneSetID = c(2, 3, 14, 23, 25)) # kinase gene sets
dev.off()

for(i in 1:length(result$Description)){
  print(result[i, "Description"])
  
  setEPS()
  filepath1 = paste("..\\results\\shap_analysis\\rankedgsea_sample11835_gseaplot2_geneset_", i, ".eps", sep='')
  postscript(filepath1)
  gseaplot2(result, geneSetID = i, title = result[i, "Description"])
  dev.off()
  
  setEPS()
  filepath2 = paste("..\\results\\shap_analysis\\rankedgsea_sample11835_gseaplot_geneset_", i, ".eps", sep='')
  postscript(filepath2)
  gseaplot(result, geneSetID = i, title = result[i, "Description"])
  dev.off()
}

setEPS()
postscript("..\\results\\shap_analysis\\rankedgsea_sample11835_emapplot.eps")
pairsim = pairwise_termsim(result)
emapplot(pairsim)
dev.off()

# https://www.genepattern.org/modules/docs/GSEA/14