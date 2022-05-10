#!/bin/bash
spark-submit rdd_chi_2.py hdfs:///user/devset.json hdfs://stopwords.txt > output_rdd.txt