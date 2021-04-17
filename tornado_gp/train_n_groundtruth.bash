#!/bin/bash
for i in {1..100}
do
   python3 training_module.py $i
done

for i in {1..100}
do
   python3 groundtruth_module.py $i
done
