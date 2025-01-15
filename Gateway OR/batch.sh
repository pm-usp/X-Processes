#!/bin/bash

python -u xprocesses.py -log input-logs/BPIC13_cp-PreProcessed-NoFilter-4act.xes.gz -isl 10 -rnd 2 -gen 5000 -tme 3600000 -con 35 -fit 1 -prc 1 -gnl 1 -smp 1 > output1.log
python -u xprocesses.py -log input-logs/BPIC13_cp-PreProcessed-NoFilter-4act.xes.gz -isl 10 -rnd 2 -gen 5000 -tme 3600000 -con 35 -fit 1 -prc 1 -gnl 1 -smp 1 > output2.log
python -u xprocesses.py -log input-logs/BPIC13_cp-PreProcessed-NoFilter-4act.xes.gz -isl 10 -rnd 2 -gen 5000 -tme 3600000 -con 35 -fit 1 -prc 1 -gnl 1 -smp 1 > output3.log
python -u xprocesses.py -log input-logs/BPIC13_cp-PreProcessed-NoFilter-4act.xes.gz -isl 40 -rnd 10 -gen 5000 -tme 3600000 -con 100 -fit 1 -prc 1 -gnl 1 -smp 1 > output-BPIC13_cp-PreProcessed-NoFilter-4act.xes.log
python -u xprocesses.py -log input-logs/BPIC13_inc-PreProcessed-NoFilter-4act.xes.gz -isl 40 -rnd 10 -gen 5000 -tme 3600000 -con 100 -fit 1 -prc 1 -gnl 1 -smp 1 > output-BPIC13_inc-PreProcessed-NoFilter-4act.xes.log
python -u xprocesses.py -log input-logs/BPIC15_1-PreProcessed-Filtered.xes.gz -isl 40 -rnd 10 -gen 5000 -tme 3600000 -con 100 -fit 1 -prc 1 -gnl 1 -smp 1 > output-BPIC15_1-PreProcessed-Filtered.xes.log
python -u xprocesses.py -log input-logs/BPIC15_2-PreProcessed-Filtered.xes.gz -isl 40 -rnd 10 -gen 5000 -tme 3600000 -con 100 -fit 1 -prc 1 -gnl 1 -smp 1 > output-BPIC15_2-PreProcessed-Filtered.xes.log
python -u xprocesses.py -log input-logs/BPIC15_3-PreProcessed-Filtered.xes.gz -isl 40 -rnd 10 -gen 5000 -tme 3600000 -con 100 -fit 1 -prc 1 -gnl 1 -smp 1 > output-BPIC15_3-PreProcessed-Filtered.xes.log
python -u xprocesses.py -log input-logs/BPIC15_4-PreProcessed-Filtered.xes.gz -isl 40 -rnd 10 -gen 5000 -tme 3600000 -con 100 -fit 1 -prc 1 -gnl 1 -smp 1 > output-BPIC15_4-PreProcessed-Filtered.xes.log
python -u xprocesses.py -log input-logs/BPIC15_5-PreProcessed-Filtered.xes.gz -isl 40 -rnd 10 -gen 5000 -tme 3600000 -con 100 -fit 1 -prc 1 -gnl 1 -smp 1 > output-BPIC15_5-PreProcessed-Filtered.xes.log
python -u xprocesses.py -log input-logs/RTFMP-PreProcessed-NoFilter.xes.gz -isl 40 -rnd 10 -gen 5000 -tme 3600000 -con 100 -fit 1 -prc 1 -gnl 1 -smp 1 > output-RTFMP-PreProcessed-NoFilter.xes.log
python -u xprocesses.py -log input-logs/SEPSIS-PreProcessed-NoFilter.xes.gz -isl 40 -rnd 10 -gen 5000 -tme 3600000 -con 100 -fit 1 -prc 1 -gnl 1 -smp 1 > output-SEPSIS-PreProcessed-NoFilter.xes.log
python -u xprocesses.py -log input-logs/BPIC12-PreProcessed-NoFilter-24act.xes.gz -isl 40 -rnd 10 -gen 5000 -tme 3600000 -con 100 -fit 1 -prc 1 -gnl 1 -smp 1 > output-BPIC12-PreProcessed-NoFilter-24act.xes.log
python -u xprocesses.py -log input-logs/BPIC14-PreProcessed-Filtered.xes.gz -isl 40 -rnd 10 -gen 5000 -tme 3600000 -con 100 -fit 1 -prc 1 -gnl 1 -smp 1 > output-BPIC14-PreProcessed-Filtered.xes.log
python -u xprocesses.py -log input-logs/BPIC17-PreProcessed-Filtered.xes.gz -isl 40 -rnd 10 -gen 5000 -tme 3600000 -con 100 -fit 1 -prc 1 -gnl 1 -smp 1 > output-BPIC17-PreProcessed-Filtered.xes.log