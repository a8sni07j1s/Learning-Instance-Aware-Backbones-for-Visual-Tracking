# ***Instance-Aware Backbones***

# content

## 1. Datasets

The list in dataset folders contains the datasets used for various network training and testing. We trained VGG16_IB/VGG16_TB, VGG19_IB/VGG19_TB, VGG-m-2048_IB/VGG-m-2048_TB, AlexNet_IB/AlexNet_TB and ResNet50_IB/ResNet50_TB, of which the feature layers are used as the tracker backbones, using ILSVRC classification 2012 dataset. We train and test trackers (with pretraining backbones), such as CREST, ECO, HDT, DAT, DSiam, SiamRPN + +, Trans-T and TrDiMP, using corresponding tracking datasets.

Note:

- IB denotes instance-backbone.
- TB denotes traditional-backbone.

## 2. Codes

The code folder contains backbone training code, tracker training, testing and evaluation code. We train VGG16, VGG19, VGG-m-2048, AlexNet and ResNet50 to obtain the pretrained backbones, which are transferred to the corresponding tracker for training, testing and evaluation. 

## 3. Models

For quick test and evaluation, the pretrained backbone models and the well-trained tracker models should be download at [Google Drive](https://drive.google.com/drive/folders/1FD0cdBoVaJAbmFywlBcDMp1yDUl3eZax?usp=sharing). The model/backbone folder contains the pretrained backbones of VGG16, VGG19, VGG-m-2048, AlexNet and ResNet50. Each network model includes a model_ IB and a model_TB.

The model/tracker folder contains the well-trained SiamRPN + +, TransT and TrDiMP tracker models; other trackers are online trained trackers.

### 3.1 Backbones

#### 3.1.1 The backbones  trained in MATLAB

The models of VGG16, VGG19, VGG-m-2048, and AlexNet  are trained in MATLAB. 

#### 3.1.2 The backbones  trained in PyTorch

The models of ResNet50, ResNet50_Siam are trained in PyTorch. 

### 3.2 Trackers

We provide the well-trained models of SiamRPN++ , TransT, and TrDiMP in the model/tracker folder.

# How to train, test and evaluate the models.

The backbone codes, tracker codes, and benchmark codes can be found in the code folder.

### Installation

These codes are run on a Ubuntu 18.4 system with Tesla V100 GPUs.

#### 1. Backbones

Backbone codes can be found in code/backbone folder.

##### 1.1 The backbones  trained in MATLAB

###### VGG16, VGG19, VGG-m-2048, AlexNet, ResNet50 

- VGG16, VGG19

  ```
  @InProceedings{Simonyan15,
    author       = "Karen Simonyan and Andrew Zisserman",
    title        = "Very Deep Convolutional Networks for Large-Scale Image Recognition",
    booktitle    = "International Conference on Learning Representations",
    year         = "2015",
  }
  ```

- VGG-m-2048

  ```
  @InProceedings{Chatfield14,
    author       = "Ken Chatfield and Karen Simonyan and Andrea Vedaldi and Andrew Zisserman",
    title        = "Return of the Devil in the Details: Delving Deep into Convolutional Nets",
    booktitle    = "British Machine Vision Conference",
    year         = "2014",
  }
  ```

- AlexNet

  ```
  @InProceedings{Krizhevsky-NIPS-2012,
  Title = {ImageNet classification with deep convolutional neural networks},
  Author = {A. Krizhevsky and I. Sutskever, G. E. Hinton},
  Booktitle = {NIPS},
  Year = {2012}
  }
  ```

- ResNet50

  ```
  @article{He2015,
  	author = {Kaiming He and Xiangyu Zhang and Shaoqing Ren and Jian Sun},
  	title = {Deep Residual Learning for Image Recognition},
  	journal = {arXiv preprint arXiv:1512.03385},
  	year = {2015}
  }
  ```

(1) Prepare training datasets

Taking AlexNet for example:

- Download [ILSVRC Classification 2012](https://image-net.org/challenges/LSVRC/2017/) dataset and put it into the Alexnet/data folder.

  - Imagenet-2012

  ```
  @article{ILSVRC15,
  Author = {Olga Russakovsky and Jia Deng and Hao Su and Jonathan Krause and Sanjeev Satheesh and Sean Ma and Zhiheng Huang and Andrej Karpathy and Aditya Khosla and Michael Bernstein and Alexander C. Berg and Li Fei-Fei},
  Title = { {ImageNet Large Scale Visual Recognition Challenge} },
  Year = {2015},
  journal   = {International Journal of Computer Vision (IJCV)},
  doi = {10.1007/s11263-015-0816-y},
  volume={115},
  number={3},
  pages={211-252}
  }
  ```


(2) Install Dependencies

MATLAB 2020b (Linux)

(3) Training

In code/backbone/VGG16, VGG19, VGG-m-2048, and AlexNet folders, the training IB backbone or TB backbone can be chosen in MATLAB/simpleNN/vl_simpleNN.m:

```
res(i+1).x = vl_nnsoftmaxloss_ib(res(i).x, l.class) ; %IB Loss 
%       res(i+1).x = vl_nnsoftmaxloss_tb(res(i).x, l.class) ; %TB Loss 
```

Then, start your training by running (taking AlexNet for example):

```
cd Alexnet/examples/imagenet/
run cnn_imagenet.m
```

##### 1.2 The backbones  trained in Pytorch

###### ResNet50_IB, ResNet50_Siam_IB （License: BSD 3-Clause "New" or "Revised" License）

(1) Install Dependence

```
conda create --name backbone_IB python=3.7
conda activate backbone_IB
conda install pytorch=1.3.0 torchvision cuda10.0 -c pytorch
```

(2) Training

- Download training dataset [ILSVRC Classification 2012](https://image-net.org/challenges/LSVRC/2017/), and put it into code/backbone/ResNet50/Python folder.

-  The training IB backbone or TB backbone can be chosen:

  ```
  criterion = CrossEntropy_with_var_norm_Loss(args.gpu, args.alpha_weight) #IB Loss
  # criterion = nn.CrossEntropyLoss().cuda(args.gpu) #TB Loss
  ```

- Single node, multiple GPUs:

  ```
  python main.py -a resnet50 --dist-url 'tcp://127.0.0.1:FREEPORT' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 [imagenet-folder with train and val folders]
  ```

- Multiple nodes:

  Node 0:

  ```
  python main.py -a resnet50 --dist-url 'tcp://IP_OF_NODE0:FREEPORT' --dist-backend 'nccl' --multiprocessing-distributed --world-size 2 --rank 0 [imagenet-folder with train and val folders]
  ```

  Node 1:

  ```
  python main.py -a resnet50 --dist-url 'tcp://IP_OF_NODE0:FREEPORT' --dist-backend 'nccl' --multiprocessing-distributed --world-size 2 --rank 1 [imagenet-folder with train and val folders]
  ```

#### 2. Trackers

The tracker codes can be found in code/tracker folder.

##### 2.1 The trackers trained in MATLAB

Online trained trackers CREST (no license), ECO (license: GNU General Public License v3.0), HDT (license: Apache License 2.0), and DSiam (no license) are trained in MATLAB (MATLAB 2020b (Linux)). These trackers are evaluated on OTB-2015, LaSOT (Apache License 2.0), and UAV123 benchmarks (See [3.Benchmark](#3.-Benchmark) for details).

- CREST

  ```
  @inproceedings{song-iccv17-CREST,
      author    = {Song, Yibing and Ma, Chao and Gong, Lijun and Zhang, Jiawei and Lau, Rynson and Yang, Ming-Hsuan}, 
      title     = {CREST: Convolutional Residual Learning for Visual Tracking}, 
      booktitle = {IEEE International Conference on Computer Vision},
      pages     = {2555 -- 2564},
      year      = {2017}
  }
  ```

- ECO

  ```
  @InProceedings{DanelljanCVPR2017,
  Title = {ECO: Efficient Convolution Operators for Tracking},
  Author = {Danelljan, Martin and Bhat, Goutam and Shahbaz Khan, Fahad and Felsberg, Michael},
  Booktitle = {CVPR},
  Year = {2017}
  }
  ```

- HDT

  ```
  @InProceedings{Qi-iccv17-HDT,
  Title = {Hedged deep tracking},
  Author = {Yuankai Qi, Shengping Zhang, Lei Qin, Hongxun Yao, Qingming Huang†, Jongwoo Lim, Ming-Hsuan Yang},
  Booktitle = {IEEE International Conference on Computer Vision},
  Year = {2017}
  }
  ```

- Dsaim

  ```
  Qing Guo, Wei Feng, Ce Zhou, Rui Huang, Liang Wan, Song Wang. Learning Dynamic Siamese Network for Visual Object Tracking. In ICCV 2017.
  ```

- OTB-2015

  ```
  @inproceedings{wu-iccv15-OTB,
      author    = {Y. Wu, J. Lim, M. Yang}, 
      title     = {Object tracking benchmark}, 
      booktitle = {IEEE Trans. Pattern Anal. Mach. Intell},
      pages     = {1834 -- 1858},
      year      = {2015}
  }
  ```

- LaSOT

  ```
  LaSOT: A High-quality Large-scale Single Object Tracking Benchmark
  H. Fan*, H. Bai*, L. Lin, F. Yang, P. Chu, G. Deng, S. Yu, Harshit, M. Huang, J Liu, Y. Xu, C. Liao, L Yuan, and H. Ling
  arXiv:2009.03465, 2020.
  LaSOT: A High-quality Benchmark for Large-scale Single Object Tracking
  H. Fan*, L. Lin*, F. Yang*, P. Chu*, G. Deng, S. Yu, H. Bai, Y. Xu, C. Liao, and H. Ling
  In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2019.
  ```

- UAV123

  ```
  @InProceedings{Mueller-eccv-uav
  Title = {A benchmark and simulator for UAV tracking},
  Author = {M. Mueller, N. Smith, and B. Ghanem},
  Booktitle = {in Comput. Vision ECCV},
  Year = {2016}
  }
  ```

##### 2.2 The trackers trained in Pytorch

######  2.2.1 SiamRPN++ (license: Apache License 2.0)

```
@inproceedings{TransT,
title={Transformer Tracking},
author={Chen, Xin and Yan, Bin and Zhu, Jiawen and Wang, Dong and Yang, Xiaoyun and Lu, Huchuan},
booktitle={CVPR},
year={2021}
}
```

(1) Install Dependence

- Create environment and activate

  ```
  conda create --name pysot python=3.7
  conda activate pysot
  ```

- Install numpy/pytorch/opencv

  ```
  conda install numpy
  conda install pytorch=0.4.1 torchvision cuda90 -c pytorch
  pip install opencv-python
  ```

- Install other requirements

  ```
  pip install pyyaml yacs tqdm colorama matplotlib cython tensorboardX
  ```

  Build extensions 

- ```
  Create environment and activate
  ```

- Try with scripts

  ```
  bash install.sh /path/to/your/conda pysot
  ```

(2) Quick test and evaluate the tracker

- Download datasets and put them into `testing_dataset` directory.  

  - [OTB2015]((http://cvlab.hanyang.ac.kr/tracker_benchmark/index.html))  

  -  [LaSOT](http://vision.cs.stonybrook.edu/~lasot/)  
  -  [UAV123](https://cemse.kaust.edu.sa/ivul/uav123)

  Download the json files of these datasets from [Google Drive](https://drive.google.com/drive/folders/10cfXjwQQBQeu48XMf2xc_W1LucpistPI) or [BaiduYun](https://pan.baidu.com/s/1js0Qhykqqur7_lNRtle1tA#list/path=%2F), and put them into the corresponding dataset folders.

- Test the tracker

  ```
  cd experiments/siamrpn_r50_l234_dwxcorr_8gpu
  python -u ../../tools/test.py 	\
  	--snapshot model/tracker/SiamRPN++/ siamrpn++_IB.pth(siamrpn++_TB.pth)	\ # model path
  	--dataset OTB100 	\ # dataset name
  	--config config.yaml	  # config file
  ```

- Evaluate the tracker

  ```
  python ../../tools/eval.py 	 \
  	--tracker_path ./results \ # result path
  	--dataset OTB100        \ # dataset name
  	--num 1 		 \ # number thread to eval
  	--tracker_prefix 'model'   # tracker_name
  ```

(3) Train, test and evaluate the tracker

- Add PySOT to your PYTHONPATH

  ```
  export PYTHONPATH=/path/to/pysot:$PYTHONPATH
  ```

- Prepare training datasets

  - [YOUTUBEBB](https://research.google.com/youtube-bb/)

    ```
    @InProceedings{RealCVPR2017,
    Title = {YouTube-BoundingBoxes: A large high-precision human-annotated data set for object detection in video},
    Author = {E. Real, J. Shlens, S. Mazzocchi, X. Pan, and V. Vanhoucke},
    Booktitle = {CVPR},
    Year = {2017}
    }
    ```

  - [VID](http://image-net.org/challenges/LSVRC/2017/)

    ```
    @InProceedings{RussakovskyCV2015,
    Title = {ImageNet large scale visual recognition challenge},
    Author = {O. Russakovsky, J. Deng, H. Su, J. Krause, S. Satheesh, S. Ma},
    Booktitle = {CV},
    Year = {2015}
    }
    ```

  - [DET](http://image-net.org/challenges/LSVRC/2017/)

    ```
    @InProceedings{RussakovskyCV2015,
    Title = {ImageNet large scale visual recognition challenge},
    Author = {O. Russakovsky, J. Deng, H. Su, J. Krause, S. Satheesh, S. Ma},
    Booktitle = {CV},
    Year = {2015}
    }
    ```

  - [COCO](http://cocodataset.org/)

    ```
    @InProceedings{Lin-eccv-2014,
    Title = {Microsoft coco: Common objects in context},
    Author = {T. Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan},
    Booktitle = {ECCV},
    Year = {2014}
    }
    ```

- Prepare pretrained backbones

  We have placed the pretrained backbones into the pysot-master/pretrained_models folder.

- Training

  ```
  cd experiments/siamrpn_r50_l234_dwxcorr_8gpu
  ```

  Multi-processing Distributed Data Parallel Training

  - Single node, multiple GPUs:

    ```
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    python -m torch.distributed.launch \
        --nproc_per_node=8 \
        --master_port=2333 \
        ../../tools/train.py --cfg config.yaml
    ```

  - Multiple nodes:

    Node 1: (IP: 192.168.1.1, and has a free port: 2333) master node

    ```
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    python -m torch.distributed.launch \
        --nnodes=2 \
        --node_rank=0 \
        --nproc_per_node=8 \
        --master_addr=192.168.1.1 \  # adjust your ip here
        --master_port=2333 \
        ../../tools/train.py
    ```

    Node 2:

    ```
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    python -m torch.distributed.launch \
        --nnodes=2 \
        --node_rank=1 \
        --nproc_per_node=8 \
        --master_addr=192.168.1.1 \
        --master_port=2333 \
        ../../tools/train.py
    ```

- Testing

  After training, you can test snapshots on the test datasets (taking OTB-2015 for example).

  ```
  python -u ../../tools/test.py 	\
  	--snapshot snapshot/checkpoint_e18.pth
  	--dataset OTB100 	\ # dataset name
  	--config config.yaml	  # config file
  ```

- Evaluation

  ```
  python ../../tools/eval.py 	 \
  	--tracker_path ./results \ # result path
  	--dataset OTB100        \ # dataset name
  	--num 4 		 \ # number thread to eval
  	--tracker_prefix 'checkpoint_e18'   # tracker_name
  ```

###### 2.2.2 TranT (license: GNU General Public License v3.0)

```
@inproceedings{TransT,
title={Transformer Tracking},
author={Chen, Xin and Yan, Bin and Zhu, Jiawen and Wang, Dong and Yang, Xiaoyun and Lu, Huchuan},
booktitle={CVPR},
year={2021}
}
```

(1) Install Dependencies

- Create and activate conda enviroment

  ```
  conda create -n transt python=3.7
  conda activate transt
  ```

- Install Pytorch

  ```
  conda install -c pytorch pytorch=1.5 torchvision=0.6.1 cudatoolkit=10.2
  ```

- Install other  packages

  ```
  conda install matplotlib pandas tqdm
  pip install opencv-python tb-nightly visdom scikit-image tikzplotlib gdown
  conda install cython scipy
  sudo apt-get install libturbojpeg
  pip install pycocotools jpeg4py
  pip install wget yacs
  pip install shapely==1.6.4.post2
  ```

- Setup the environment

  Create the default environment setting files.

  ```
  # Change directory to <PATH_of_TransT>
  cd TransT
  
  # Environment settings for pytracking. Saved at pytracking/evaluation/local.py
  python -c "from pytracking.evaluation.environment import create_default_local_file; create_default_local_file()"
  
  # Environment settings for ltr. Saved at ltr/admin/local.py
  python -c "from ltr.admin.environment import create_default_local_file; create_default_local_file()"
  ```

You can modify these files to set the paths to datasets, results paths etc.

- Add the project path to environment variable.

  ```
  export PYTHONPATH=<path_of_TransT>:$PYTHONPATH
  ```

(2) Quick test and evaluate the tracker

- Download datasets and put them into testing_dataset directory.  

  - [OTB2015]((http://cvlab.hanyang.ac.kr/tracker_benchmark/index.html))  

  -  [LaSOT](http://vision.cs.stonybrook.edu/~lasot/)  
  -  [UAV123](https://cemse.kaust.edu.sa/ivul/uav123)

  Download json files in  [PySOT](https://github.com/STVIR/pysot) or [here](https://drive.google.com/file/d/1PItNIOkui0iGCRglgsZPZF1-hkmj7vyv/view?usp=sharing) to use PYSOT-toolkit for evaluation, and put them into the corresponding dataset folders.

- Test the tracker

  - You need to specify the path of the tracker model and dataset in  TransT-main/pysot_toolkit/test.py . 

    ```
    net_path = 'model/tracker/TransT/TransT_IB.pth(TransT_TB.pth)' #Absolute path of the model
    dataset_root= 'path_to_datasets' #Absolute path of the datasets
    ```
  
    Also, you need to specify the path of the pretrained backbone in TransT-main/ltr/models/backbone/resnet.py.

    ```
    pretrained_model = 'model/backbone/ResNet50/IB_model/ResNet_Pytorch/resnet50_IB.pth' #resnet50_IB.pth
    # pretrained_model = 'model/backbone/ResNet50/TB_model/ResNet_Pytorch/resnet50_TB.pth' #resnet50_TB.pth
    ```
  
  - Then run the following commands.

    ```
    conda activate TransT
    cd TransT
    python -u pysot_toolkit/test.py --dataset <name of dataset> --name 'transt' #test tracker #test tracker
    ```
  
- Evaluate the tracker

  ```
  python pysot_toolkit/eval.py --tracker_path results/ --dataset <name of dataset> --num 1 --tracker_prefix 'transt' #eval tracker
  ```

(3)  Train, test and evaluate the tracker

- Prepare training dataset

  Modify ltr/admin/local.py to set the datasets paths, results paths etc.

  - [GOT-10k](http://got-10k.aitestunion.com/index)

    ```
    @ARTICLE{8922619,
      author={Huang, Lianghua and Zhao, Xin and Huang, Kaiqi},
      journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
      title={GOT-10k: A Large High-Diversity Benchmark for Generic Object Tracking in the Wild}, 
      year={2021},
      volume={43},
      number={5},
      pages={1562-1577},
      doi={10.1109/TPAMI.2019.2957464}}
    ```

  - [TrackingNet](https://tracking-net.org/)

    ```
    @InProceedings{Muller_2018_ECCV,
    author = {Muller, Matthias and Bibi, Adel and Giancola, Silvio and Alsubaihi, Salman and Ghanem, Bernard},
    title = {TrackingNet: A Large-Scale Dataset and Benchmark for Object Tracking in the Wild},
    booktitle = {The European Conference on Computer Vision (ECCV)},
    month = {September},
    year = {2018}
    }	
    ```

  - [COCO2017](https://cocodataset.org/)

  - [LaSOT](http://vision.cs.stonybrook.edu/~lasot/)

- Training

  - The installation script will automatically generate a local configuration file "admin/local.py". In case the file was not generated, run `admin.environment.create_default_local_file()` to generate it. Next, set the paths to the training workspace, i.e. the directory where the checkpoints will be saved. Also set the paths to the datasets you want to use. 

  - You need to specify the path of the backbone model  in the TransT-main/ltr/models/backbone/resnet.py.

    ```
    pretrained_model = 'model/backbone/ResNet50/IB_model/ResNet_Pytorch/resnet50_IB.pth' #resnet50_IB.pth
    # pretrained_model = 'model/backbone/ResNet50/TB_model/ResNet_Pytorch/resnet50_TB.pth' #resnet50_TB.pth
    ```

  -  If all the dependencies have been correctly installed, you can train a network using the run_training.py script in the correct conda environment.

    ```
    conda activate transt
    cd TransT/ltr
    python run_training.py transt transt
    ```

- Testing

  - You need to specify the path of the tracker model and dataset in the /TransT-main/pysot_toolkit/test.py . 

    ```
    net_path = 'TransT-main/checkpoints/ltr/transt/transt/TransT_ep*.pth.tar' #Absolute path of the model
    dataset_root= 'path_to_datasets' #Absolute path of the datasets
    ```

  - Also, you need to specify the path of the pretrained backbone in TransT-main/ltr/models/backbone/resnet.py.

    ```
    pretrained_model = 'model/backbone/ResNet50/IB_model/ResNet_Pytorch/resnet50_IB.pth' #resnet50_IB.pth
    # pretrained_model = 'model/backbone/ResNet50/TB_model/ResNet_Pytorch/resnet50_TB.pth' #resnet50_IB.pths
    ```

  - Then run the following commands.

    ```
    conda activate TransT
    cd TransT
    python -u pysot_toolkit/test.py --dataset <name of dataset> --name 'transt' #test tracker #test tracker
    ```

- Evaluation

  ```
  python pysot_toolkit/eval.py --tracker_path results/ --dataset <name of dataset> --num 1 --tracker_prefix 'transt' #eval tracker
  ```

###### 2.2.3 TrDiMP (license: MIT License)

```
@inproceedings{Wang_2021_Transformer,
    title={Transformer Meets Tracker: Exploiting Temporal Context for Robust Visual Tracking},
    author={Wang, Ning and Zhou, Wengang and Wang, Jie and Li, Houqiang},
    booktitle={The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2021}
}
```

(1) Install Dependence

- Create and activate a conda environment

  ```
  conda create --name pytracking python=3.7
  conda activate pytracking
  ```

- Install PyTorch

  ```
  conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
  ```

  **Note:**

  - It is possible to use any PyTorch supported version of CUDA (not necessarily v10).
  - For more details about PyTorch installation, see https://pytorch.org/get-started/previous-versions/.

- Install matplotlib, pandas, tqdm, opencv, scikit-image, visdom, tikzplotlib, gdown, and tensorboad

  ```
  conda install matplotlib pandas tqdm
  pip install opencv-python visdom tb-nightly scikit-image tikzplotlib gdown
  ```

- Install the coco toolkit

  If you want to use COCO dataset for training, install the coco python toolkit. You additionally need to install cython to compile the coco toolkit.

  ```
  conda install cython
  pip install pycocotools
  ```

- Install ninja-build for Precise ROI pooling

  ```
  sudo apt-get install ninja-build
  ```
  
- Install jpeg4py

  ```
  sudo apt-get install libturbojpeg
  pip install jpeg4py 
  ```
  
- Setup the environment

  Create the default environment setting files.

  ```
  # Environment settings for pytracking. Saved at pytracking/evaluation/local.py
  python -c "from pytracking.evaluation.environment import create_default_local_file; create_default_local_file()"
  
  # Environment settings for ltr. Saved at ltr/admin/local.py
  python -c "from ltr.admin.environment import create_default_local_file; create_default_local_file()"
  ```


(2) Quick test and evaluate the tracker

- Download datasets and put them into testing_dataset directory.  

  - [OTB2015]((http://cvlab.hanyang.ac.kr/tracker_benchmark/index.html))  

  -  [LaSOT](http://vision.cs.stonybrook.edu/~lasot/)  
  -  [UAV123](https://cemse.kaust.edu.sa/ivul/uav123)

- <a name="Testing">Test the tracker</a>

  You need to specify the path of the tracker model in the TransformerTrack-main/pytracking/parameter/trdimp/trdimp.py . :

  ```
  params.net = NetWithBackbone(net_path='model/tracker/TrDiMP/trdimp_IB.pth', use_gpu=params.use_gpu)
  ```

  Test the tracker on certain dataset.

  ```
  python run_tracker.py trdimp trdimp --dataset_name dataset_name  --debug debug --threads threads
  ```

  Here, the dataset_name is the name of the dataset used for evaluation, e.g. `otb`. See TransformerTrack-main/pytracking/evaluation/dataset.py for the list of datasets which are supported. The `debug` parameter can be used to control the level of debug visualizations. `threads` parameter can be used to run on multiple threads.

- <a name="Evaluation">Evaluate  the tracker</a>

  Put the test results into pysot-master/experiments/siamrpn_r50_l234_dwxcorr_8gpu, and run the following commands:

  ```
  cd experiments/siamrpn_r50_l234_dwxcorr_8gpu
  python ../../tools/eval.py 	 \
  	--tracker_path ./results \ # result path
  	--dataset OTB100        \ # dataset name
  	--num 1 		 \ # number thread to eval
  	--tracker_prefix 'trdimp'   # tracker_name
  ```

(3) Train, test and evaluate the tracker

- Prepare testing datasets

  - [GOT-10k](http://got-10k.aitestunion.com/index)
  - [TrackingNet](https://tracking-net.org/)
  - [COCO2017](https://cocodataset.org/)
  - [LaSOT](http://vision.cs.stonybrook.edu/~lasot/)

- Training

  - You need to specify the path of the pretrained backbone in  TransformerTrack-main/ltr/models/backbone/resnet.py.
  
    ```
    pretrained_model = 'model/backbone/ResNet50/IB_model/ResNet_Pytorch/resnet50_IB.pth' #resnet50_IB.pth
    # pretrained_model = 'model/backbone/ResNet50/TB_model/ResNet_Pytorch/resnet50_TB.pth' #resnet50_IB.pth
    ```

  - The installation script will automatically generate a local configuration file "admin/local.py". In case the file was not generated, run `admin.environment.create_default_local_file()` to generate it. Next, set the paths to the training workspace, i.e. the directory where the checkpoints will be saved. Also set the paths to the datasets you want to use. If all the dependencies have been correctly installed, you can train the network using the run_training.py script in the correct conda environment.

    ```
    conda activate pytracking
    python run_training.py trdimp trdimp
    ```
  
- Testing

  See <a href="#Testing">Test the tracker</a>.

- Evaluation

  See <a href="#Evaluation">Evaluate the tracker</a>.

#### 3. Benchmark

We evaluate our tracking results on OTB benchmark, UAV123 benchmark, and LaSOT benchmark,. We provide modified benchmark codes in the code/benchmark.

###### 3.1 OTB benchmark

```
@inproceedings{wu-iccv17-OTB,
    author    = {Y. Wu, J. Lim, M. Yang}, 
    title     = {Object tracking benchmark}, 
    booktitle = {IEEE Trans. Pattern Anal. Mach. Intell},
    pages     = {1834 -- 1858},
    year      = {2015}
}
```

(1) Install Dependence

MATLAB 2020b (Linux)

(2) Evaluation and Plotting

```
run mian_runing.m
```

###### 3.2 UAV123 benchmark

```
@InProceedings{Mueller-eccv-uav
Title = {A benchmark and simulator for UAV tracking},
Author = {M. Mueller, N. Smith, and B. Ghanem},
Booktitle = {in Comput. Vision ECCV},
Year = {2016}
}
```

(1) Install Dependence

MATLAB 2020b (Linux)

(2) Evaluation and Plotting

```
run mian_runing.m
```

###### 3.3 LaSOT benchmark 

```
LaSOT: A High-quality Large-scale Single Object Tracking Benchmark
H. Fan*, H. Bai*, L. Lin, F. Yang, P. Chu, G. Deng, S. Yu, Harshit, M. Huang, J Liu, Y. Xu, C. Liao, L Yuan, and H. Ling
arXiv:2009.03465, 2020.
LaSOT: A High-quality Benchmark for Large-scale Single Object Tracking
H. Fan*, L. Lin*, F. Yang*, P. Chu*, G. Deng, S. Yu, H. Bai, Y. Xu, C. Liao, and H. Ling
In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2019.
```

(1) Install Dependence

MATLAB 2020b (Linux)

(2) Evaluation and Plotting

```
run run_tracker_performance_evaluation.m
```















