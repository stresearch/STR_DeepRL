# Suggested Papers for Week 4 - Also as a Send Off Gift

## LLM / NLP

### [Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/pdf/2203.02155.pdf)
This is the GPT RLHF paper. A great example of RL used in the wild, and an important paper for understanding how modern LLMs are trained

> Summary:
The paper introduces InstructGPT, a method for aligning large language models with user intent by fine-tuning them with human feedback. Using a dataset of labeler-written and OpenAI API submitted prompts, the authors fine-tune GPT-3 with supervised learning and further refine it with reinforcement learning based on rankings of model outputs. The resulting InstructGPT model, despite having significantly fewer parameters, outperforms the GPT-3 model in human evaluations, showing improvements in truthfulness and a reduction in toxic output.


### [RLPROMPT: Optimizing Discrete Text Prompts with Reinforcement Learning](https://arxiv.org/pdf/2205.12548.pdf)
This is another clever application of RL to LLMs, this time learning prompt optimization. A good look at using RL to solve an interesting large combinatoric problem while gaining insight into LLMs in the process. Authored by Eric Xing (an ML bigshot) and Zhiting Hu (a potential STR collaborator)


> Summary:
RLPROMPT is a method that optimizes discrete text prompts using reinforcement learning. It introduces a policy network for generating desired discrete prompts efficiently after training with rewards. The paper highlights the limitations of existing prompt optimization methods and showcases the superior performance of RLPROMPT across various types of LMs and tasks, even when the optimized prompts appear as ungrammatical gibberish.


### [Reasoning with Language Model is Planning with World Model](https://arxiv.org/pdf/2305.14992.pdf)
Not actually RL, so maybe we can leave this one out, but it's super interesting--they use a strcutured reasoning framework to get LLMs to plan via MCTS. Seems like the next natural step is to embed RL into this somewhere--a discussion of how a researcher could extend this work into RL might be a good discussion for the class. ALso features two potential STR collaborators, Zhiting Hu and Daisy Zhe Wang.

> Summary:
This research proposes a new framework called RAP (Reasoning via Planning) for overcoming LLMs' limitation in tasks like action plan generation and complex reasoning. RAP utilizes LLMs as both a world model and a reasoning agent, incorporating a principled planning algorithm for strategic exploration in the vast reasoning space. The LLMs build a reasoning tree under the guidance of task-specific rewards, demonstrating superior performance over various strong baselines in challenging reasoning problems.

### [Is Reinforcement Learning (Not) for Natural Language Processing?: Benchmarks, Baselines, and Building Blocks for Natural Language Policy Optimization](https://arxiv.org/pdf/2210.01241v2.pdf)
Might be too long and comprehensive to go over in too much detail, but a very interesting exploration of the application of RL to NLP

> Summary:
The paper introduces an open-source library, RL4LMs, to optimize language generators using reinforcement learning. It discusses the challenges of using RL for text generation due to the combinatorial action space and presents a benchmark, GRUE, for evaluating RL algorithms for NLP tasks. The paper also introduces NLPO, an RL algorithm that effectively reduces the combinatorial action space in language generation, showing better performance and stability compared to previous methods.



### [Learning to Model the World with Language](https://arxiv.org/pdf/2308.01399.pdf)
Discussion of Dynalang, which Avinash presented to RLRG


> Summary:
The paper presents Dynalang, an agent that integrates language understanding with future prediction as a self-supervised learning objective. Dynalang predicts future text and image representations and learns to act from imagined model rollouts, using language to enhance task performance in various settings. The agent leverages diverse types of language to improve its understanding and action predictions in different environments.

[GPT-4V(ision) system card](https://openai.com/research/gpt-4v-system-card)

> Summary:
The abstract discusses the introduction of GPT-4 with Vision (GPT-4V), which allows users to command GPT-4 to analyze image inputs. This enhancement is part of an effort in the field of artificial intelligence to integrate various modalities like image inputs into large language models (LLMs), extending their capabilities and applications. The text outlines the potential for multimodal LLMs to tackle new tasks and offer unique experiences. The authors specifically delve into the safety aspects of GPT-4V, examining the evaluations, preparations, and mitigation efforts tailored for image inputs, building upon the existing safety work done for GPT-4.

[Training diffusion models with reinforcement learning](https://arxiv.org/pdf/2305.13301)
Black, Kevin, et al. "Training diffusion models with reinforcement learning." arXiv preprint arXiv:2305.13301 (2023).


> Summary:
This research investigates the use of reinforcement learning for optimizing diffusion models based on downstream objectives such as image quality or drug effectiveness. It introduces denoising diffusion policy optimization (DDPO), a more effective policy gradient algorithm compared to alternative reward-weighted likelihood approaches. DDPO adapts text-to-image diffusion models to objectives that are challenging to express via prompting, improving prompt-image alignment using feedback from a vision-language model without additional data or human annotation.

## Offline / Off-policy RL
I am less certain that these papers are worthy and interesting, but wanted to include some advances in this space due to the survey feedback. I would love to hear what you think of these and if you have different ideas

### [Behavior Proximal Policy Optimization](https://arxiv.org/pdf/2302.11312.pdf)
Applying offline learning to PPO


> Summary: 
This paper presents Behavior Proximal Policy Optimization (BPPO), an algorithm designed for offline reinforcement learning (RL) without requiring extra constraints or regularization compared to PPO. The authors found that some on-policy algorithms can naturally solve offline RL problems due to their inherent conservatism. Through extensive experimentation, BPPO showcased superior performance against other state-of-the-art offline RL algorithms on the D4RL benchmark.

### [Harnessing Mixed Offline Reinforcement Learning Datasets via Trajectory Weighting](https://arxiv.org/pdf/2306.13085.pdf)
By reweighting an offline trajectory dataset and retraining, they achieve better results in learning from the dataset in a way that's agnostic to the offline RL training algorithm used

> Summary: 
The study addresses the issue of underutilization of high-return trajectories in mixed datasets. By re-weighting the dataset sampling, an artificial dataset is induced, which enhances the performance of various offline RL algorithms. Empirical results demonstrate enhanced performance using this strategy, including when combined with algorithms like CQL, IQL, and TD3+BC.


[A Closer Look at Offline RL Agents](https://proceedings.neurips.cc/paper_files/paper/2022/file/3908cadfcc99db12001eafb1207353e9-Paper-Conference.pdf)
Fu, Yuwei, Di Wu, and Benoit Boulet. "A closer look at offline rl agents." Advances in Neural Information Processing Systems 35 (2022): 8591-8604.

> Summary: 
This research offers insight into the behavior of offline RL agents, unveiling that more performant agents might learn lower-quality representations and inaccurate value functions. The proposed experimental setups help identify offline RL agents' bottlenecks. A new offline RL algorithm, designed based on the evaluation results, is introduced, which achieves state-of-the-art performance.

[Pre-Training for Robots: Offline RL Enables Learning New Tasks in a Handful of Trials](https://arxiv.org/pdf/2210.05178)
Kumar, Aviral, et al. "Pre-training for robots: Offline rl enables learning new tasks from a handful of trials." arXiv preprint arXiv:2210.05178 (2022).


> Summary: 
This paper presents a framework (PTR) for robots based on offline RL, enabling the learning of new tasks with minimal demonstrations by utilizing pre-training on existing datasets and rapid fine-tuning on a new task. PTR, using a conservative Q-learning (CQL) approach, showcases the ability to learn new tasks with as few as 10 demonstrations.

[Cal-QL: Calibrated Offline RL Pre-Training for Efficient Online Fine-Tuning](https://arxiv.org/pdf/2303.05479)
Nakamoto, Mitsuhiko, et al. "Cal-ql: Calibrated offline rl pre-training for efficient online fine-tuning." arXiv preprint arXiv:2303.05479 (2023).


> Summary: 
The paper presents Cal-QL, a method for obtaining effective policy initialization from offline data, which also enables efficient online fine-tuning. By learning a conservative value function initialization, Cal-QL outperforms other methods in numerous fine-tuning benchmark tasks.


[A Survey on Offline Reinforcement Learning: Taxonomy, Review, and Open Problems](https://ieeexplore.ieee.org/iel7/5962385/6104215/10078377.pdf)
Prudencio, Rafael Figueiredo, Marcos ROA Maximo, and Esther Luna Colombini. "A survey on offline reinforcement learning: Taxonomy, review, and open problems." IEEE Transactions on Neural Networks and Learning Systems (2023).


> Summary: 
This comprehensive survey provides a taxonomy for classifying offline RL methods and a review of the latest algorithmic breakthroughs and existing benchmarks. The paper highlights the challenges and future research directions in the rapidly growing field of offline RL.

## Imitation Learning

[Hierarchical Model-Based Imitation Learning for Planning in Autonomous Driving](https://arxiv.org/pdf/2210.09539)
Bronstein, Eli, et al. from Waymo "Hierarchical model-based imitation learning for planning in autonomous driving." 2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2022.


> Summary: 
The research demonstrates a large-scale application of model-based generative adversarial imitation learning (MGAIL) for urban self-driving. By utilizing a hierarchical model and combining MGAIL losses with open-loop behavior cloning losses, the authors achieve a steerable policy for robust navigation in real-world driving scenarios, approaching the performance of experts.

[A Survey on Imitation Learning Techniques for End-to-End Autonomous Vehicles](https://ieeexplore.ieee.org/abstract/document/9700770)
Le Mero, Luc, et al. "A survey on imitation learning techniques for end-to-end autonomous vehicles." IEEE Transactions on Intelligent Transportation Systems 23.9 (2022): 14128-14147.


> Summary:
This survey reviews imitation learning techniques for autonomous vehicles, categorizing the literature into Behavioral Cloning, Direct Policy Learning, and Inverse Reinforcement Learning. The paper provides a comprehensive overview of current literature and datasets, along with identifying future research directions to advance imitation learning-based systems for autonomous vehicles.

## MARL and Meta-RL
These were the ones you provided, so I haven't had a chance to look into them too deeply but I'm planning to include them

### [MAESTRO: Open-Ended Environment Design for Multi-Agent Reinforcement Learning](https://arxiv.org/pdf/2303.03376.pdf)

> Summary:
The paper introduces MAESTRO, an extension of Unsupervised Environment Design (UED) for multi-agent environments, focusing on two-player zero-sum settings. MAESTRO considers the interplay between the environment and co-player to shape a curriculum in multi-agent domains, leading to the efficient production of adversarial joint curricula over both environments and co-players. The experimental results highlight MAESTRO's superior performance over various baselines in competitive two-player games.

### [Human-Timescale Adaptation in an Open-Ended Task Space](https://arxiv.org/pdf/2301.07608.pdf)

> Summary:
This work presents an RL agent that can rapidly adapt to novel 3D problems, similar to human adaptation capabilities. The agent, named AdA, employs meta-reinforcement learning, a large-scale attention-based memory architecture, and an automated curriculum to enable efficient knowledge exploitation and hypothesis-driven exploration. The results demonstrate scalable performance across a broad range of tasks, providing a foundation for the development of more adaptive and general RL agents.

[Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](http://proceedings.mlr.press/v70/finn17a/finn17a.pdf)
Finn, Chelsea, Pieter Abbeel, and Sergey Levine. "Model-agnostic meta-learning for fast adaptation of deep networks." International conference on machine learning. PMLR, 2017.

> Summary:
The paper proposes a model-agnostic meta-learning algorithm compatible with any model trained with gradient descent. The algorithm trains a model across various learning tasks to facilitate fast adaptation to new tasks with minimal training data. The approach enhances the model's ability to be easily fine-tuned, showing advanced performance on few-shot image classification and other benchmarks.



[Stabilizing Unsupervised Environment Design with a Learned Adversary](https://arxiv.org/abs/2308.10797)
Mediratta, Ishita, et al. "Stabilizing Unsupervised Environment Design with a Learned Adversary." arXiv preprint arXiv:2308.10797 (2023).

> Summary:
The research explores the challenges of PAIRED, a pioneering approach for Unsupervised Environment Design (UED), and offers solutions to enhance its practical performance. The proposed improvements enable PAIRED to outperform state-of-the-art methods, yielding more robust agents capable of handling various complex, procedurally-generated environments.


[Deep Reinforcement Learning with Plasticity Injection](https://arxiv.org/pdf/2305.15555)
Nikishin, Evgenii, et al. "Deep Reinforcement Learning with Plasticity Injection." arXiv preprint arXiv:2305.15555 (2023).

> Summary:
This paper introduces plasticity injection, an intervention to increase network plasticity in deep RL without altering the trainable parameters. Plasticity injection helps in diagnosing and overcoming performance plateaus due to a lack of plasticity, enhancing the computational efficiency and performance of RL training in various environments.


[GNM: A General Navigation Model to Drive Any Robot](https://arxiv.org/pdf/2210.03370.pdf)
Shah, Dhruv, et al. "Gnm: A general navigation model to drive any robot." 2023 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2023.

> Abstract: Learning provides a powerful tool for vision-based navigation, but the capabilities of learning-based policies are constrained by limited training data. If we could combine data from all available sources, including multiple kinds of robots, we could train more powerful navigation models. In this paper, we study how a general goal-conditioned model for vision-based navigation can be trained on data obtained from many distinct but structurally similar robots, and enable broad generalization across environments and embodiments. We analyze the necessary design decisions for effective data sharing across robots, including the use of temporal context and standardized action spaces, and demonstrate that an omnipolicy trained from heterogeneous datasets outperforms policies trained on any single dataset. We curate 60 hours of navigation trajectories from 6 distinct robots, and deploy the trained GNM on a range of new robots, including an underactuated quadrotor. We find that training on diverse data leads to robustness against degradation in sensing and actuation. Using a pre-trained navigation model with broad generalization capabilities can bootstrap applications on novel robots going forward, and we hope that the GNM represents a step in that direction. For more information on the datasets, code, and videos, please check out our project page this https [URL](https://sites.google.com/view/drive-any-robot).

> Summary:
The paper presents a general goal-conditioned model for vision-based navigation trained on data from diverse robots. The omnipolicy trained from this heterogeneous data outperforms policies trained on single datasets, demonstrating robustness against sensory and actuation degradation and offering a foundation for developing navigation models for novel robots.


[META-REINFORCEMENT LEARNING: ALGORITHMS AND APPLICATIONS](https://stacks.stanford.edu/file/druid:zf342ty7446/final_thesis-augmented.pdf)

> Summary:
The thesis focuses on meta-reinforcement learning, discussing the importance of effectively utilizing limited interaction episodes for learning new tasks. It introduces a new algorithm called Dream, which decouples task learning and the use of limited interaction shots, showing promising results in various applications, including language learning and automated grading.


[A Survey of Meta-Reinforcement Learning](https://arxiv.org/pdf/2301.08028)
Beck, Jacob, et al. "A survey of meta-reinforcement learning." arXiv preprint arXiv:2301.08028 (2023).

> Summary:
This survey provides a comprehensive overview of meta-RL, detailing its problem setting and major variations. It discusses meta-RL research based on task distribution and learning budget for each task, offering insight into meta-RL algorithms and applications, and presenting open problems that need addressing for enhancing the usability of meta-RL in deep RL practices.

## Hierarchical RL

[Hierarchical Planning Through Goal-Conditioned Offline Reinforcement Learning](https://arxiv.org/pdf/2205.11790)
Li, Jinning, et al. "Hierarchical planning through goal-conditioned offline reinforcement learning." IEEE Robotics and Automation Letters 7.4 (2022): 10216-10223.

> Summary:
In this study, the authors tackle the challenge of applying Offline Reinforcement Learning (RL) to tasks that extend over a longer time frame. They introduce a hierarchical planning model that integrates a low-level goal-conditioned RL policy with a high-level goal planner. A key feature is their strategy for dealing with out-of-distribution goals by altering the goal sampling method. The high-level planner determines intermediate objectives using model-based planning techniques and forecasts sub-goal sequences based on the value function learned by the low-level policy. A Conditional Variational Autoencoder is employed to sample relevant high-dimensional sub-goal options. When tested in long-horizon driving and robot navigation challenges, their approach surpasses other hierarchical models and standard planners.


[Multi-Stage Cable Routing through Hierarchical Imitation Learning](https://arxiv.org/abs/2307.08927)
Luo, Jianlan, et al. "Multi-Stage Cable Routing through Hierarchical Imitation Learning." arXiv preprint arXiv:2307.08927 (2023).


> Summary:
This paper delves into the intricacies of teaching robots to carry out multi-stage manipulation tasks, exemplified by the task of guiding a cable through multiple clips. Given the complexities of manipulating flexible objects, relying on visual feedback, and navigating numerous steps, conventional learning methods face difficulties in ensuring high success rates. Addressing these challenges, the authors introduce an imitation learning system that benefits from vision-based policies learned from demonstrations. This method is applied to both motor control (low-level) and sequencing (high-level) stages. The system is adept at adjusting its approach based on the situation, whether it means choosing a different controller, retrying, or taking corrective actions. When applied to the cable routing task, the system showcases impressive adaptability to various clip placements.

[Hierarchical Reinforcement Learning: A Survey and Open Research Challenges](https://www.mdpi.com/2504-4990/4/1/9)
Hutsebaut-Buysse, Matthias, Kevin Mets, and Steven Latré. "Hierarchical reinforcement learning: A survey and open research challenges." Machine Learning and Knowledge Extraction 4.1 (2022): 172-221.

> Summary: 
Reinforcement Learning (RL) empowers agents to make sequential decisions through interaction with their environment. For highly intricate environments, the typical random exploration approach can be inefficient. Hierarchical reinforcement learning (HRL) addresses this by using temporal and state abstractions, making the behavior of RL systems more reusable and understandable. This survey commences by discussing specific task-centric techniques that offer insights into the utilization of custom-designed abstractions. The discourse then shifts to the Options framework, a versatile approach enabling semi-automatic discovery and learning of abstractions. Following this, the authors discuss the goal-conditional approach that embeds sub-behaviors in a continuous space. The paper culminates by spotlighting potential avenues for future research, particularly regarding HRL agents that can learn from interactions within complex, high-dimensional environments.