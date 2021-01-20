## $5 Tech Unlocked 2021!
[Buy and download this product for only $5 on PacktPub.com](https://www.packtpub.com/)
-----
*The $5 campaign         runs from __December 15th 2020__ to __January 13th 2021.__*

# TensorFlow 2 Reinforcement Learning Cookbook

<a href="https://www.packtpub.com/product/tensorflow-2-reinforcement-learning-cookbook/9781838982546?utm_source=github&utm_medium=repository&utm_campaign=9781838982546"><img src="https://static.packt-cdn.com/products/9781838982546/cover/smaller" alt="TensorFlow 2 Reinforcement Learning Cookbook" height="256px" align="right"></a>

This is the code repository for [TensorFlow 2 Reinforcement Learning Cookbook](https://www.packtpub.com/product/tensorflow-2-reinforcement-learning-cookbook/9781838982546?utm_source=github&utm_medium=repository&utm_campaign=9781838982546), published by Packt.

**Over 50 recipes to help you build, train, and deploy learning agents for real-world applications**

## What is this book about?
With deep reinforcement learning, you can build intelligent agents, products, and services that can go beyond computer vision or perception to perform actions. TensorFlow 2.x is the latest major release of the most popular deep learning framework used to develop and train deep neural networks (DNNs). This book contains easy-to-follow recipes for leveraging TensorFlow 2.x to develop artificial intelligence applications.

Starting with an introduction to the fundamentals of deep reinforcement learning and TensorFlow 2.x, the book covers OpenAI Gym, model-based RL, model-free RL, and how to develop basic agents. You'll discover how to implement advanced deep reinforcement learning algorithms such as actor-critic, deep deterministic policy gradients, deep-Q networks, proximal policy optimization, and deep recurrent Q-networks for training your RL agents. As you advance, you’ll explore the applications of reinforcement learning by building cryptocurrency trading agents, stock/share trading agents, and intelligent agents for automating task completion. Finally, you'll find out how to deploy deep reinforcement learning agents to the cloud and build cross-platform apps using TensorFlow 2.x.

By the end of this TensorFlow book, you'll be able to:
| |
| :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Build**: Deep RL agents from scratch using the all-new and powerful TensorFlow 2.x framework and Keras API |
| **Implement**: Deep RL algorithms (DQN, A3C, DDPG, PPO, SAC etc.) with minimal lines of code |
| **Train**: Deep RL agents in simulated environments (gyms) beyond toy-problems and games to perform real-world tasks like cryptocurrency trading, stock trading, tweet/email management and more! |
| **Scale**: Distributed training of RL agents using TensorFlow 2.x, Ray + Tune + RLLib |
| **Deploy**: RL agents to the cloud and edge for real-world testing by creating cloud services, web apps and Android mobile apps using TensorFlow Lite, TensorFlow.js, ONNX and Triton |
| |

This book covers the following exciting features: 
* Build deep reinforcement learning agents from scratch using the all-new TensorFlow 2.x and Keras API
* Implement state-of-the-art deep reinforcement learning algorithms using minimal code
* Build, train, and package deep RL agents for cryptocurrency and stock trading
* Deploy RL agents to the cloud and edge to test them by creating desktop, web, and mobile apps and cloud services
* Speed up agent development using distributed DNN model training
* Explore distributed deep RL architectures and discover opportunities in AIaaS (AI as a Service)

If you feel this book is for you, get your [copy](https://www.amazon.com/dp/183898254X) today!

<a href="https://www.packtpub.com/?utm_source=github&utm_medium=banner&utm_campaign=GitHubBanner"><img src="https://raw.githubusercontent.com/PacktPublishing/GitHub/master/GitHub.png" alt="https://www.packtpub.com/" border="5" /></a>


## Instructions and Navigations
All of the code is organized into folders. For example,

The code will look like the following:
```
class Agent(object):
    def __init__(self, action_dim=5, input_dim=(1, 8 * 8)):
       self.brain = Brain(action_dim, input_dim)
       self.policy = DiscretePolicy(action_dim)
    def get_action(self, obs):
       action_logits = self.brain.process(obs)
       action = self.policy.get_action(np.squeeze(action_logits, 0))
    return action
```

**Following is what you need for this book:**
The code in this book is extensively tested on Ubuntu 18.04 and Ubuntu 20.04 and should work with later versions of Ubuntu if Python 3.6+ is available. With Python 3.6+ installed along with the necessary Python packages, as listed at the start of each of the recipes, the code should run fine on Windows and macOS X too. 

It is advised to create and use a Python virtual environment named tfrl-cookbook to install the packages and run the code in this book. A Miniconda or Anaconda installation for Python virtual environment management is recommended. Follow these steps to set up a virtual environment:

1. Install system dependencies:`sudo apt install -y make cmake ffmpeg`
2. Install miniconda: `wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && bash -b -p ${HOME}/miniconda3`
3. Setup conda python virtual environment: `bash && conda env create -f tfrl-cookbook.yml -n "tfrl-cookbook"`
   You are all set!
4. Activate the `tfrl-cookbook` conda python environment: `conda activate tfrl-cookbook` and get started with the recipes in the book!

It is highly recommended to star and fork the GitHub repository to receive updates and improvements to the code recipes.We urge you to share what you build and also engage with other readers and the community [here](https://github.com/PacktPublishing/Tensorflow-2-Reinforcement-Learning-Cookbook/discussions).

With the following software and hardware list you can run all code files present in the book (Chapter 1-9).

### Software and Hardware List

| Chapter  | Software required                                       | OS required                        |
| -------- | --------------------------------------------------------| -----------------------------------|
| 1 - 9    | Python 3.6 (or later versions)                          | Windows, Mac OS X, and Linux (Any) |
|   9      | Android Studio                                          | Windows, Mac OS X, and Linux (Any) |

We also provide a PDF file that has color images of the screenshots/diagrams used in this book. [Click here to download it](https://static.packt-cdn.com/downloads/9781838982546_ColorImages.pdf).


### Related products <Other books you may enjoy>
* Mastering Reinforcement Learning with Python [[Packt]](https://www.packtpub.com/product/mastering-reinforcement-learning-with-python/9781838644147?utm_source=github&utm_medium=repository&utm_campaign=9781838644147) [[Amazon]](https://www.amazon.com/dp/1838644148)

* Deep Reinforcement Learning with Python - Second Edition [[Packt]](https://www.packtpub.com/product/deep-reinforcement-learning-with-python-second-edition/9781839210686?utm_source=github&utm_medium=repository&utm_campaign=9781839210686) [[Amazon]](https://www.amazon.com/dp/1839210680)

## Get to Know the Author
**Praveen Palanisamy** works on advancing AI for autonomous systems as a senior AI engineer at Microsoft. In the past, he has developed AI algorithms for autonomous vehicles using deep reinforcement learning, and has worked with start-ups and in academia to build autonomous robots and intelligent systems. He is the inventor of more than 15 patents on learning-based AI systems. He is the author of *HOIAWOG: Hands-On Intelligent Agents with OpenAI Gym*, which provides a step-by-step guide to developing deep RL agents to solve complex problems from scratch. He has a master's in robotics from Carnegie Mellon University.


