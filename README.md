# Learning to Discover Skills through Guidance (DISCO-DANCE)


This repository is an official PyTorch implementation of the paper, DISCO-DANCE: Learning to Discover Skills through Guidance, NeurIPS 2023. This codebase was adapted from [URLB](https://github.com/rll-research/url_benchmark).

[arXiv](https://arxiv.org/abs/2310.20178) / [project page](https://mynsng.github.io/discodance/) / [poster](https://drive.google.com/file/d/1OzvaebPbe8RIMW8FrH5RdbDLF0QGbX8Z/view)

Authors: [Hyunseung Kim](https://mynsng.github.io/)\*, Byungkun Lee\*, [Hojoon Lee](https://joonleesky.github.io/about/), Dongyoon Hwang, Sejik Park, Kyushik Min, and [Jaegul Choo](https://sites.google.com/site/jaegulchoo/).

## Requirements
We assume you have access to a GPU that can run CUDA 11.1 and CUDNN 8. Then, the simplest way to install all required dependencies is to create an anaconda environment by running
```sh
conda env create -f requirements.yaml
```
After the instalation ends you can activate your environment with
```sh
conda activate DISCO-DANCE
```

## Instructions
### Pre-training
To run pre-training use the `pretrain_maze.py` script
```sh
python pretrain_maze.py maze_type=square_bottleneck
```
This script will produce several agent snapshots after training for `1M`, `2M`, ..., `5M` frames. The snapshots will be stored under the following directory:
```sh
./models/<agent_name>/<maze_type>/
```

### Monitoring
The console output is also available in a form:
```
| train | F: 6000 | S: 3000 | E: 6 | L: 1000 | R: 5.5177 | FPS: 96.7586 | T: 0:00:42
```
a training entry decodes as
```
F  : total number of environment frames
S  : total number of agent steps
E  : total number of episodes
R  : episode return
FPS: training throughput (frames per second)
T  : total training time
```

## Citations

```
@article{kim2024learning,
  title={Learning to discover skills through guidance},
  author={Kim, Hyunseung and LEE, BYUNG KUN and Lee, Hojoon and Hwang, Dongyoon and Park, Sejik and Min, Kyushik and Choo, Jaegul},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  year={2024}
}
```

## Contact

For personal communication, please contact Hyunseung Kim, or Byungkun Lee at 

`{mynsng, byungkun.lee}@kaist.ac.kr`.