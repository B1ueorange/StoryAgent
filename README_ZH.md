# 介绍

StoryAgent是一个能将用户创造的剧本生成视频的深度学习模型工具。用户通过我们提供的工具组合，进行剧本创作、剧照生成、图片/视频生成等工作。

StoryAgent的模型由[ModelScope](https://github.com/modelscope/modelscope)开源模型社区提供支持。


# 功能特性

- 剧本生成（Script Generation）
  - 用户指定故事主题和背景，即可生成剧本
  - 剧本生成模型基于LLM（如GPT 3.5），可生成多种风格的剧本
  - 用户可以与agent持续交流对话，从而调整剧本内容
- 剧照生成（Movie still Generation）
  - 通过输入一幕剧本，即可生成对应的剧照场景图片
  - 提供prompt助手功能，帮助用户引导模型生成高质量图片
- 视频生成（Video Generation）
  - 图生视频
  - 支持高分辨率视频生成
- 音乐生成（Music Generation）
  - 自定义风格的背景音乐



# 快速开始

## 兼容性验证

已经验证过的环境：

- python3.8


## 安装指南

### conda虚拟环境

使用conda虚拟环境，参考[Anaconda](https://docs.anaconda.com/anaconda/install/)来管理您的依赖，安装完成后，执行如下命令：

```shell
conda create -n motion_agent python=3.8
conda activate motion_agent

GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/modelscope/motionagent.git --depth 1
cd motionagent

# 安装依赖
pip3 install -r requirements.txt

# 运行应用
python3 app.py
```

## 模型列表

[1]  Qwen-Turbo

[2]  SDXL 1.0

[3]  I2VGen-XL

[4]  MusicGen

[5]  ChatGPT

[6]  DALL-E

[7]  Wanx-v1

# License

This project is licensed under the [Apache License (Version 2.0)](https://github.com/modelscope/modelscope/blob/master/LICENSE).

