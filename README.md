# Amazon Review Analysis with SPARK

In this project we analyze amazon review data. This script is meant to run on a SPARK cluster. It will use the chi-square test to determine the words for each category that are the most unique ones. 

To run the RDD pipeline, start the bash-file `run_mr_rdd.sh`. The script requires an input file and a file that contains the stopwords. You can change the file-paths if you need. 
The result will be written into `output_rdd.txt`.

The second analysis uses the PySpark API instead of RDD. It is located in the `ml_kernel.ipynb` notebook. This should train a model that takes a review and predicts their category. However, during my access to the cluster, it was not able to run any ML-tasks without errors. Therefore, this might not work. 
To use the code, just copy the notebook into you PySpark environment.