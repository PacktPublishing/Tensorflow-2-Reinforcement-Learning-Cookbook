FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
# Deploying Deep RL Agents to the cloud
# Chapter 7, TensorFlow2.x Reinforcement Learning Cookbook | Praveen Palanisamy
LABEL maintainer="emailid@domain.tld"

RUN apt-get install -y wget git make cmake zlib1g-dev && rm -rf /var/lib/apt/lists/*

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN apt-get update

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

CMD [ "5_packaging_rl_agents_for_deployment.py" ]