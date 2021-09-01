## Program-Enhanced-Table-Fact-Checking
Code for the EMNLP 2020 paper "Program Enhanced Fact Verification with Verbalization and Graph Attention Network"

## Environment
```
  python  3.7.3
  pytorch 1.3.1
```

## Data Preparation
Please refer to the data preprocessing in [TABFACT baselines](https://github.com/wenhuchen/Table-Fact-Checking) to get the following data.

0. The preprocessed data for LPA 
```
$program_dir$
```
1. The preprocessed data for Table-BERT

Note that we take the horizontal version.
```
$table_dir$
```

## A. Program Selection
In './program_selection', please run 'run_BERT_margin_loss.py' for program selection.

You can always train your own program selection model, and get the selected programs for the next verbalization step.

To simplify this procedure, we provide a [checkpoint](https://drive.google.com/file/d/1EBwMm6zMBmRDqS9-R3n9FTQRwHt4THsa/view?usp=sharing) of program selection model so that you can just load it and get the selected programs for verbalization:
```
python run_BERT_margin_loss.py --do_gen --load_dir $the downloaded ckpt$ --data_dir $program_dir$
```

## B. Verbalization with Program Execution
In './verbalization', please run 'run_verbalize_attn.py' for verbalization.

```
python run_verbalize_attn.py --selected_prog_dir $the output of prev program selection$ --table_bert_dir $table_dir$ --save_root_dir $determine your verb output path$
```

## C. Graph-based Verification Network
To train the grah-based verification model (here we can load a table-bert baseline ckpt to accelerate the training process):
```
python run_gvn.py --do_train --do_eval --tune_table_after10k --load_dir $table-bert baseline ckpt$ --data_dir $table_dir$ --data_dir_verb $verb output path$
```

To load the [checkpoint](https://drive.google.com/file/d/1B3URYBbDu_ybZEwk-NsNMex7F3sggeMK/view?usp=sharing) and only run eval:
```
python run_gvn.py --do_eval --do_test --do_simple_test --do_complex_test --do_small_test --load_dir_whole $the downloaded ckpt$ --data_dir $table_dir$ --data_dir_verb $verb output path$
```


## Contact

For any questions, please send email to authors.




