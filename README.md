# cyc-cp
This repository provides the benchmark for Cycling Close Pass Near Miss  (Cyc-CP).


## Installation
1. Clone the repo: `git clone https://github.com/SustainableMobility/cyc-cp.git`
2. Install the package: 
    1. `cd cyc-cp`
    2. `pip install -e .`
## How To Use
1. `cd cyc-cp`
2. Train crnn or i3d model following:
    1. `python ./cnm/scene_lvl/i3d/i3d.py --csv_data_path path/to/csv_data --image_data_path path/to/image_data --exp_data_dir path/to/save/exp_data`   
    2. `python ./cnm/scene_lvl/crnn/crnn.py --csv_data_path path/to/csv_data --image_data_path path/to/image_data --exp_data_dir path/to/save/exp_data` 
    * where the meaning of the arguments can be found in the code help. Specifically, 
        * --csv_data_path: The file path of the .csv file with dataset info.
        * --image_data_path: The directory path saving all video frames.
        * --exp_data_dir: The directory to save results to.

## Dataset Preparation
* Victorian On-bike Cycling (legacy): available on [Monash Bridges](https://figshare.com/projects/A_Benchmark_for_Cycling_Close_Pass_Near_Miss_Event_Detection_from_Video_Streams/163438)
* Victorian On-bike Cycling (ongoing): under collection ...
* CARLA (simulation): available on [Monash Bridges](https://figshare.com/projects/A_Benchmark_for_Cycling_Close_Pass_Near_Miss_Event_Detection_from_Video_Streams/163438)
* NuScences: available on <https://www.nuscenes.org/>


## Hardware Requirements Summary
* Disk: to save all datasets about > 2TB disk space is required.
* RAM and GPU: (only tested on Victorian On-bike Cycling (legacy))
    * Scene-level:
        * I3D: (Batch_size:16, image_size: 256x342, frames: [-5, 15])
            * GPU memory: 7.5GB, CPU memory: 3 GB (RTX-2080 has 8 GB memory, so that’s why [-5, 15] frames are included in a video clip.)
        * CRNN: (Batch_size: 16, image_size: 224x224, frame: [20,25])
            * GPU memory: 3 GB, CPU memory: 6 GB
    * Instance-level (**TODO**)

More Notes about the project can be found in [the shared google doc](https://docs.google.com/document/d/13UxCa-qcIuyZ6V3-oTSfUFmnSvWxZjIalBQ6MdcVUCs/edit?usp=sharing).
