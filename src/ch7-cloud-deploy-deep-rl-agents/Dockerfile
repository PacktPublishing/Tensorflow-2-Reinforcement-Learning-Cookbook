FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
# TensorFlow2.x Reinforcement Learning Cookbook
# Chapter 7: Deploying Deep RL Agents to the cloud
LABEL maintainer="emailid@domain.tld"

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN apt-get update

RUN apt-get install -y wget git make cmake zlib1g-dev && rm -rf /var/lib/apt/lists/*

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 

# conda>4.9.0 is required for `--no-capture-output`
RUN conda update -n base conda

ADD . /root/tf-rl-cookbook/ch7

WORKDIR /root/tf-rl-cookbook/ch7

RUN conda env create -f "tfrl-cookbook.yml" -n "tfrl-cookbook"

ENTRYPOINT [ "conda", "run", "--no-capture-output", "-n", "tfrl-cookbook", "python" ]

CMD [ "2_packaging_rl_agents_for_deployment.py" ]