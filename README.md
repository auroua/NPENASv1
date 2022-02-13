# Neural Predictor Guided Evolution for Neural Architecture Search

This repository contains code for paper [NPENAS: Neural Predictor Guided Evolution for Neural Architecture Search](https://arxiv.org/abs/2003.12857).

If you use the code please cite our paper.

    @article{Wei2020NPENASNP,
        title={NPENAS: Neural Predictor Guided Evolution for Neural Architecture Search},
        author={Chen Wei and Chuang Niu and Yiping Tang and Ji-min Liang},
        journal={ArXiv},
        year={2020},
        volume={abs/2003.12857}
    }

## Prerequisites
* Python 3.7
* Pytorch 1.3 
* Tensorflow 1.14.0
* ptflops `pip install --upgrade git+https://github.com/sovrasov/flops-counter.pytorch.git`
* torch-scatter `pip install torch-scatter==1.4.0`
* torch-sparse `pip install torch-sparse==0.4.3`
* torch-cluster `pip install torch-cluster==1.4.5`
* torch-spline-conv `pip install torch-spline-conv==1.1.1`

## Searching Environment
* Ubuntu 18.04
* cuda 10.0
* cudnn 7.5.1

## Usage
### Clone this project
```bash
git clone https://github.com/auroua/NPENASv1
cd NPENASv1
```
### Closed Domain Search
###### NASBench-101 Dataset 
1. Down load `NASBench-101` dataset first. We only use the `nasbench_only108.tfrecord` file.
2. Modify the `tf_records_path` variable in `nas_lib/config.py` to store the absolute path of `nasbench_only108.tfrecord`.
3. You can test the default sampling pipeline via running the following command.  *Change the `save_dir` to your directory before running.*
```bash
# gpus: the number of gpus used to execute searching.
# save_dir: the output path.
python train_multiple_gpus_close_domain.py --trials 600 --search_budget 150 --search_space nasbench_case1 --algo_params nasbench101_case1 --gpus 1 --save_dir /home/albert_wei/Disk_A/train_output_2021/npenas_101/ --comparison_type algorithm --record_full_data F
```  
4. You can test the new sampling pipeline via running the following command. *Change the `save_dir` to your directory before running.*
```bash
# gpus: the number of gpus used to execute searching.
# save_dir: the output path.
python train_multiple_gpus_close_domain.py --trials 600 --search_budget 150 --search_space nasbench_case2 --algo_params nasbench101_case2 --gpus 1 --save_dir /home/albert_wei/Disk_A/train_output_2021/npenas_101/ --comparison_type algorithm --record_full_data F
``` 
5. Run the following command to visualize the comparison of algorithms. *Change the `save_dir` to the path of `save_dir` in step 3 or 4.*
```bash
python tools_close_domain/visualize_results.py --search_space nasbench_101 --draw_type ERRORBAR --save_dir /home/albert_wei/Disk_A/train_output_npenas/close_domain_case1/ 
```
**Visualize the results of comparing algorithms on NAS-Bench-101 search space via using the new sampling pipeline.**

<p align="center">
  <img width="576" height="432" src="/images/NASBench-101.jpg">
</p>

6. Scaling Factor Analysis
```bash
python train_multiple_gpus_close_domain.py --trials 200 --search_budget 150 --search_space nasbench_case2 --algo_params nasbench101_case2 --gpus 1 --save_dir /home/albert_wei/Disk_A/train_output_2021/npenas_101/ --comparison_type scalar_compare --record_full_data T --record_kt T 
```

7. ReLU CELU Analysis
```bash
python train_multiple_gpus_close_domain.py --trials 200 --search_budget 150 --search_space nasbench_case2 --algo_params nasbench101_case2 --gpus 1 --save_dir /home/albert_wei/Disk_A/train_output_2021/npenas_101/ --comparison_type relu_celu --record_full_data T --record_kt T --relu_celu_comparison_algo_type NPENAS_NP
```


###### NASBench-201 Dataset 
We only compared algorithms using the new sampling pipeline on the NASBench-201 dataset. 
1. Down load the `NASBench-201` dataset first. In this experiment, we use the `NASBench-201` dataset with version `v1_1-096897`, and the file name is `NAS-Bench-201-v1_1-096897.pth`.
2. Modify the `nas_bench_201_path` variable in `nas_lib/config.py` to store the absolute path of `NAS-Bench-201-v1_1-096897.pth`. 
3. As the `NASBench-201` dataset is too large. and the `NASBench-201` utilizes edges as operates and nodes as features. In order to use the `NAS-Bench-201` dataset, 
we have to convert this dataset first.  
4. Modify the variable `nas_bench_201_converted_path` in `nas_lib/config.py` to the path that store the processed `NAS-Bench-201` dataset.
5. Run the following command to execute convert. **This step is memory consuming. The memory of my computer is 32G.**
```bash
# dataset: the dataset used to train the architectures in NASBench-201, choices: ['cifar10-valid', 'cifar100', 'ImageNet16-120']
python tools_close_domain/train_init_dataset.py --dataset cifar10-valid
```
6. You can run the following command to compare algorithms on `NAS-Bench-201` dataset. *Change the `save_dir` to your directory before running.*
```bash
# gpus: the number of gpus used to execute searching.
# save_dir: the output path.
python train_multiple_gpus_close_domain.py --trials 600 --search_budget 100 --search_space nasbench_201 --algo_params nasbench_201 --gpus 1 --multiprocessing-distributed False --save_dir /home/albert_wei/Disk_A/train_output_npenas/npenas_201/ --comparison_type algorithm --record_full_data F --dataset cifar100
``` 

7. Run the following command to visualize the comparison of algorithms. *Change the `save_dir` to the path of `save_dir` in step 6.*
```bash
python tools_close_domain/visualize_results.py --search_space nasbench_201 --save_dir /home/albert_wei/Disk_A/train_output_npenas/npenas_201/ --draw_type ERRORBAR
```

8. Scaling Factor Analysis
```bash
python train_multiple_gpus_close_domain.py --trials 200 --search_budget 100 --search_space nasbench_201 --algo_params nasbench_201 --gpus 1 --save_dir /home/albert_wei/Disk_A/train_output_2021/npenas_201/ --comparison_type scalar_compare --record_full_data T --record_kt T --dataset cifar100
```

9. ReLU CELU Analysis
```bash
python train_multiple_gpus_close_domain.py --trials 200 --search_budget 100 --search_space nasbench_201 --algo_params nasbench_201 --gpus 1 --save_dir /home/albert_wei/Disk_A/train_output_2021/npenas_201/ --comparison_type relu_celu --record_full_data T --record_kt T --relu_celu_comparison_algo_type NPENAS_NP --dataset cifar100
```

###### NASBench-NLP Dataset 
We only compared algorithms using the new sampling pipeline on the NASBench-201 dataset. 
1. Down load the `NASBench-NLP` dataset first.
2. Modify the `nas_bench_nlp_path` variable in `nas_lib/config.py` to store the folder directory of `NASBench_NLP`.
3. You can run the following command to compare algorithms on `NAS-Bench-NLP` dataset. *Change the `save_dir` to your directory before running.*
```bash
# gpus: the number of gpus used to execute searching.
# save_dir: the output path.
python train_multiple_gpus_close_domain.py --trials 600 --search_budget 100 --search_space nasbench_nlp --algo_params nasbench_nlp --gpus 1 --multiprocessing-distributed False --save_dir /home/albert_wei/Disk_A/train_output_npenas/nasbench_nlp/ --comparison_type algorithm --record_full_data F
```
4. Run the following command to visualize the comparison of algorithms. *Change the `save_dir` to the path of `save_dir` in step 6.*
```bash
python tools_close_domain/visualize_results.py --search_space nasbench_201 --save_dir /home/albert_wei/Disk_A/train_output_npenas/npenas_201/ --draw_type ERRORBAR
```
5. Scaling Factor Analysis
```bash
python train_multiple_gpus_close_domain.py --trials 200 --search_budget 100 --search_space nasbench_nlp --algo_params nasbench_nlp --gpus 1 --save_dir /home/albert_wei/Disk_A/train_output_2021/nasbench_nlp/ --comparison_type scalar_compare --record_full_data T --record_kt T
```
6. ReLU CELU Analysis
```bash
python train_multiple_gpus_close_domain.py --trials 200 --search_budget 100 --search_space nasbench_nlp --algo_params nasbench_nlp --gpus 1 --save_dir /home/albert_wei/Disk_A/train_output_2021/nasbench_nlp/ --comparison_type relu_celu --record_full_data T --record_kt T --relu_celu_comparison_algo_type NPENAS_BO
```

###### NASBench-ASR Dataset 
We only compared algorithms using the new sampling pipeline on the NASBench-201 dataset. 
1. Down load the `NASBench-ASR` dataset first.
2. Modify the `nas_bench_asr_path` variable in `nas_lib/config.py` to store the folder directory of `NASBench_ASR`.
3. You can run the following command to compare algorithms on `NAS-Bench-ASR` dataset. *Change the `save_dir` to your directory before running.*
```bash
# gpus: the number of gpus used to execute searching.
# save_dir: the output path.
python train_multiple_gpus_close_domain.py --trials 600 --search_budget 100 --search_space nasbench_asr --algo_params nasbench_asr --gpus 1 --multiprocessing-distributed False --save_dir /home/albert_wei/Disk_A/train_output_npenas/npenas_asr/ --comparison_type algorithm --record_full_data F
```
4. Run the following command to visualize the comparison of algorithms. *Change the `save_dir` to the path of `save_dir` in step 6.*
```bash
python tools_close_domain/visualize_results.py --search_space nasbench_asr --save_dir /home/albert_wei/Disk_A/train_output_npenas/npenas_201/ --draw_type ERRORBAR
```
5. Scaling Factor Analysis
```bash
python train_multiple_gpus_close_domain.py --trials 200 --search_budget 100 --search_space nasbench_asr --algo_params nasbench_asr --gpus 1 --save_dir /home/albert_wei/Disk_A/train_output_2021/npenas_asr/ --comparison_type scalar_compare --record_full_data T --record_kt T
```
6. ReLU CELU Analysis
```bash
python train_multiple_gpus_close_domain.py --trials 200 --search_budget 100 --search_space nasbench_asr --algo_params nasbench_asr --gpus 1 --save_dir /home/albert_wei/Disk_A/train_output_2021/npenas_asr/ --comparison_type relu_celu --record_full_data T --record_kt T --relu_celu_comparison_algo_type NPENAS_BO
```

### Open Domain Search
1. Run the following command to search architecture in `DARTS` search space via algorithm `NPENAS-BO`. *Change the `save_dir` to your directory before running.*
```bash
python train_multiple_gpus_open_domain.py --gpus 1 --algorithm gin_uncertainty_predictor --budget 150 --save_dir /home/albert_wei/Disk_A/train_output_npenas/npenas_open_domain_darts_1/
```

2. Run the following command to search architecture in `DARTS` search space via algorithm `NPENAS-NP`. *Change the `save_dir` to your directory before running.*
```bash
python train_multiple_gpus_open_domain.py --gpus 1 --algorithm gin_predictor --budget 100 --save_dir /home/albert_wei/Disk_A/train_output_npenas/npenas_open_domain_darts_2/
```

3. Run the following command to rank the searched architectures, and select the best to retrain. Replace `model_path` with the real searched architectures' output path.
```bash
python tools_open_domain/rank_searched_darts_arch.py --model_path /home/albert_wei/Disk_A/train_output_npenas/npenas_open_domain_darts_2/model_pkl/
```

4. Retrain the selected architecture using the following command. 
```bash
model_name: the id of select architecture
save_dir: the output path
python tools_open_domain/train_darts_cifar10.py --seed 1 --model_name ace85b6b1618a4e0ebdc0db40934f2982ac57a34ec9f31dcd8d209b3855dce1f.pkl  --save_dir /home/albert_wei/Disk_A/train_output_npenas/npenas_open_domain_darts_2/
```

5. Test the retrained architecture with the following command
```bash
model_name: the id of select architecture
save_dir: set with the save_dir in step 4
model_path: set with the save_dir in step 1 or 2 
python tools_open_domain/test_darts_cifar10.py  --model_name xxxx --save_dir xxxx  --model_path xxxx
```

6. Run the following command to visualize the normal cell and reduction cell of the searched best architecture.
```bash
python tools_open_domain/visualize_results.py --model_path xxxx --model_name xxxx
```

If you encounter the following problem please reference this link [possible deadlock in dataloader](https://github.com/pytorch/pytorch/issues/1355)
```
RuntimeError: unable to open shared memory object </torch_31124_2696026929> in read-write mode
```

**Visualize the normal cell and the reduction cell searched by `NPENAS-NP`, and this architecture achieves a testing error `2.44%`.**

![searched_architecture](/images/npenas_normal_reduction_cell.png)


You can download the best architecture's genotype file from [genotype](https://pan.baidu.com/s/1Cu2tC-6LOHQC2sYnPZkRXg) with extract code `itw9`. The address of the retrained weight file is [ckpt](https://pan.baidu.com/s/1Mef-LCstyON5b59PZOmL1A) with extract code `t9xq`.
You can use the command in step 5 to verify the model.

### Compare the different evaluation strategies
There are two mutation strategies: one-parent-one-child, one-parent-multiple-children. *Change the `save_dir` to your directory before running.*
1. Run the following command.
```bash
python train_multiple_gpus_close_domain.py --trials 600 --search_budget 150 --search_space nasbench_case1 --algo_params evaluation_compare --gpus 1 --save_dir /home/albert_wei/Disk_A/train_output_npenas/evolutionary_compare/
```
2. Visualize the results. *Set the `save_dir` with the `save_dir` in step 1.*
```bash
python tools_close_domain/visualize_results.py --search_space evaluation_compare --draw_type ERRORBAR  --save_dir /home/albert_wei/Disk_A/train_output_npenas/evolutionary_compare/
```

### Compare the paths distribution of different sampling pipeline
There are two different architecture sampling pipelines: the default sampling pipeline and the new sampling pipeline. 
Run the following code to compare the paths distribution of different sampling pipelines:
```bash
python tools_close_domain/visualize_sample_distribution.py --sample_num 5000 --seed 98765
```

### Compare the prediction performance of different methods
1. Compare the four method mentioned in the paper.
```bash
python tools_close_domain/prediction_compare.py --trials 300 --seed 434 --search_space nasbench_case1 --save_path /home/albert_wei/Disk_A/train_output_npenas/prediction_compare/prediction_compare.pkl
```
2. Parse the results generate from the above step.
```bash
python tools_close_domain/prediction_compare_parse.py --save_path /home/albert_wei/Disk_A/train_output_npenas/prediction_compare/prediction_compare.pkl
```


## Experiment Results
|  Experiment      | visualization script* | link     | password    |
| :-------------: | :----------: | :----------: | :-----------: |
|  npenas close domain search |  tools_close_domain/visualize_results.py   | [link](https://pan.baidu.com/s/1Kgc_li4oELDARA9NziHFeQ)   | k3iq    |
|  scaling factor analysis |  tools_close_domain/visualize_results_scaling_factor.py  |[link](https://pan.baidu.com/s/1oEsvSVhWfKuOYz-Et1c0Gg)   |  qd1s   |
|  relu celu comparison |       tools_close_domain/visualize_results_relu_celu.py     |[link](https://pan.baidu.com/s/1XQczk2oFbxSwWFmMxXiOsA)   |  pwvk   |
|  search space correlation analysis |     tools_ss_analysis/search_space_analysis_correlation.py       |[link](https://pan.baidu.com/s/17p_Ub9mriDD9_x2s24VkRw)   |   ebrh  |
|  search space distance distribution analysis |      tools_ss_analysis/search_space_analysis_dist_distribution.py     |[link](https://pan.baidu.com/s/17p_Ub9mriDD9_x2s24VkRw)   |  h63o   |
|  statistical testing |        tools_ss_analysis/stats_ttest.py     |[link](https://pan.baidu.com/s/1dev4bbix6BNjpRZULIS0pg)   |  h63o   |
|  mutation strategy analysis |      tools_close_domain/visualizee_results_nasbench_nlp_mutation_strategy.py     |[link](https://pan.baidu.com/s/17xQDO35mnSHuDQYr9DxDMg)   |  pn7s   |

<sub> * modify the parameters of the visualization script to view results. </sub>


## Acknowledge
1. [bananas](https://github.com/naszilla/bananas)
2. [naszilla](https://github.com/naszilla/naszilla)
3. [pytorch_geometric](https://github.com/rusty1s/pytorch_geometric)
4. [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark)
5. [detectron2](https://github.com/facebookresearch/detectron2)
6. [NAS-Bench-101](https://github.com/google-research/nasbench)
7. [NAS-Bench-201](https://github.com/D-X-Y/NAS-Bench-201)
8. [darts](https://github.com/quark0/darts)
9. [AlphaX](https://github.com/linnanwang/AlphaX-NASBench101)
10. [NAS-Bench-NLP](https://github.com/fmsnew/nas-bench-nlp-release)
11. [NAS-Bench-ASR](https://github.com/SamsungLabs/nb-asr)

## Contact
Chen Wei

email: weichen_3@stu.xidian.edu.cn