# Suggested papers for Week 4

## LLM / NLP
This is the largest category by far, for a few reasons: 
1. student interest (driven by JJ)
2. research community interest--LLMs are a big new research toy
3. opportunity to highlight the work of potential STR collaborators

### [Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/pdf/2203.02155.pdf)
This is the GPT RLHF paper. A great example of RL used in the wild, and an important paper for understanding how modern LLMs are trained

> Abstract: Making language models bigger does not inherently make them better at following a user's intent. For example, large language models can generate outputs that are untruthful, toxic, or simply not helpful to the user. In other words, these models are not aligned with their users. In this paper, we show an avenue for aligning language models with user intent on a wide range of tasks by fine-tuning with human feedback. Starting with a set of labeler-written prompts and prompts submitted through the OpenAI API, we collect a dataset of labeler demonstrations of the desired model behavior, which we use to fine-tune GPT-3 using supervised learning. We then collect a dataset of rankings of model outputs, which we use to further fine-tune this supervised model using reinforcement learning from human feedback. We call the resulting models InstructGPT. In human evaluations on our prompt distribution, outputs from the 1.3B parameter InstructGPT model are preferred to outputs from the 175B GPT-3, despite having 100x fewer parameters. Moreover, InstructGPT models show improvements in truthfulness and reductions in toxic output generation while having minimal performance regressions on public NLP datasets. Even though InstructGPT still makes simple mistakes, our results show that fine-tuning with human feedback is a promising direction for aligning language models with human intent.

or 

> Summary:
The paper introduces InstructGPT, a method for aligning large language models with user intent by fine-tuning them with human feedback. Using a dataset of labeler-written and OpenAI API submitted prompts, the authors fine-tune GPT-3 with supervised learning and further refine it with reinforcement learning based on rankings of model outputs. The resulting InstructGPT model, despite having significantly fewer parameters, outperforms the GPT-3 model in human evaluations, showing improvements in truthfulness and a reduction in toxic output.


### [RLPROMPT: Optimizing Discrete Text Prompts with Reinforcement Learning](https://arxiv.org/pdf/2205.12548.pdf)
This is another clever application of RL to LLMs, this time learning prompt optimization. A good look at using RL to solve an interesting large combinatoric problem while gaining insight into LLMs in the process. Authored by Eric Xing (an ML bigshot) and Zhiting Hu (a potential STR collaborator)


> Abstract: Prompting has shown impressive success in enabling large pretrained language models (LMs) to perform diverse NLP tasks, especially when only few downstream data are available. Automatically finding the optimal prompt for each task, however, is challenging. Most existing work resorts to tuning soft prompt (e.g., embeddings) which falls short of interpretability, reusability across LMs, and applicability when gradients are not accessible. Discrete prompt, on the other hand, is difficult to optimize, and is often created by "enumeration (e.g., paraphrasing)-then-selection" heuristics that do not explore the prompt space systematically. This paper proposes RLPrompt, an efficient discrete prompt optimization approach with reinforcement learning (RL). RLPrompt formulates a parameter-efficient policy network that generates the desired discrete prompt after training with reward. To overcome the complexity and stochasticity of reward signals by the large LM environment, we incorporate effective reward stabilization that substantially enhances the training efficiency. RLPrompt is flexibly applicable to different types of LMs, such as masked (e.g., BERT) and left-to-right models (e.g., GPTs), for both classification and generation tasks. Experiments on few-shot classification and unsupervised text style transfer show superior performance over a wide range of existing finetuning or prompting methods. Interestingly, the resulting optimized prompts are often ungrammatical gibberish text; and surprisingly, those gibberish prompts are transferrable between different LMs to retain significant performance, indicating LM prompting may not follow human language patterns.


or 

> Summary:
RLPROMPT is a method that optimizes discrete text prompts using reinforcement learning. It introduces a policy network for generating desired discrete prompts efficiently after training with rewards. The paper highlights the limitations of existing prompt optimization methods and showcases the superior performance of RLPROMPT across various types of LMs and tasks, even when the optimized prompts appear as ungrammatical gibberish.


### [Reasoning with Language Model is Planning with World Model](https://arxiv.org/pdf/2305.14992.pdf)
Not actually RL, so maybe we can leave this one out, but it's super interesting--they use a strcutured reasoning framework to get LLMs to plan via MCTS. Seems like the next natural step is to embed RL into this somewhere--a discussion of how a researcher could extend this work into RL might be a good discussion for the class. ALso features two potential STR collaborators, Zhiting Hu and Daisy Zhe Wang.

> Abstract: Large language models (LLMs) have shown remarkable reasoning capabilities, especially when prompted to generate intermediate reasoning steps (e.g., Chain-of-Thought, CoT). However, LLMs can still struggle with problems that are easy for humans, such as generating action plans for executing tasks in a given environment, or performing complex math, logical, and commonsense reasoning. The deficiency stems from the key fact that LLMs lack an internal world model to predict the world state (e.g., environment status, intermediate variable values) and simulate long-term outcomes of actions. This prevents LLMs from performing deliberate planning akin to human brains, which involves exploring alternative reasoning paths, anticipating future states and rewards, and iteratively refining existing reasoning steps. To overcome the limitations, we propose a new LLM reasoning framework, R⎯⎯⎯easoning via⎯⎯P⎯⎯⎯lanning (RAP). RAP repurposes the LLM as both a world model and a reasoning agent, and incorporates a principled planning algorithm (based on Monto Carlo Tree Search) for strategic exploration in the vast reasoning space. During reasoning, the LLM (as agent) incrementally builds a reasoning tree under the guidance of the LLM (as world model) and task-specific rewards, and obtains a high-reward reasoning path efficiently with a proper balance between exploration vs. exploitation. We apply RAP to a variety of challenging reasoning problems including plan generation, math reasoning, and logical inference. Empirical results on these tasks demonstrate the superiority of RAP over various strong baselines, including CoT and least-to-most prompting with self-consistency. RAP on LLAMA-33B surpasses CoT on GPT-4 with 33% relative improvement in a plan generation setting.


or

> Summary:
This research proposes a new framework called RAP (Reasoning via Planning) for overcoming LLMs' limitation in tasks like action plan generation and complex reasoning. RAP utilizes LLMs as both a world model and a reasoning agent, incorporating a principled planning algorithm for strategic exploration in the vast reasoning space. The LLMs build a reasoning tree under the guidance of task-specific rewards, demonstrating superior performance over various strong baselines in challenging reasoning problems.

### [Is Reinforcement Learning (Not) for Natural Language Processing?: Benchmarks, Baselines, and Building Blocks for Natural Language Policy Optimization](https://arxiv.org/pdf/2210.01241v2.pdf)
Might be too long and comprehensive to go over in too much detail, but a very interesting exploration of the application of RL to NLP

> Abstract: We tackle the problem of aligning pre-trained large language models (LMs) with human preferences. If we view text generation as a sequential decision-making problem, reinforcement learning (RL) appears to be a natural conceptual framework. However, using RL for LM-based generation faces empirical challenges, including training instability due to the combinatorial action space, as well as a lack of open-source libraries and benchmarks customized for LM alignment. Thus, a question rises in the research community: is RL a practical paradigm for NLP?
To help answer this, we first introduce an open-source modular library, RL4LMs (Reinforcement Learning for Language Models), for optimizing language generators with RL. The library consists of on-policy RL algorithms that can be used to train any encoder or encoder-decoder LM in the HuggingFace library (Wolf et al. 2020) with an arbitrary reward function. Next, we present the GRUE (General Reinforced-language Understanding Evaluation) benchmark, a set of 6 language generation tasks which are supervised not by target strings, but by reward functions which capture automated measures of human preference.GRUE is the first leaderboard-style evaluation of RL algorithms for NLP tasks. Finally, we introduce an easy-to-use, performant RL algorithm, NLPO (Natural Language Policy Optimization)} that learns to effectively reduce the combinatorial action space in language generation. We show 1) that RL techniques are generally better than supervised methods at aligning LMs to human preferences; and 2) that NLPO exhibits greater stability and performance than previous policy gradient methods (e.g., PPO (Schulman et al. 2017)), based on both automatic and human evaluations.


or

> Summary:
The paper introduces an open-source library, RL4LMs, to optimize language generators using reinforcement learning. It discusses the challenges of using RL for text generation due to the combinatorial action space and presents a benchmark, GRUE, for evaluating RL algorithms for NLP tasks. The paper also introduces NLPO, an RL algorithm that effectively reduces the combinatorial action space in language generation, showing better performance and stability compared to previous methods.



### [Learning to Model the World with Language](https://arxiv.org/pdf/2308.01399.pdf)
Discussion of Dynalang, which Avinash presented to RLRG

> Abstract: To interact with humans in the world, agents need to understand the diverse types of language that people use, relate them to the visual world, and act based on them. While current agents learn to execute simple language instructions from task rewards, we aim to build agents that leverage diverse language that conveys general knowledge, describes the state of the world, provides interactive feedback, and more. Our key idea is that language helps agents predict the future: what will be observed, how the world will behave, and which situations will be rewarded. This perspective unifies language understanding with future prediction as a powerful self-supervised learning objective. We present Dynalang, an agent that learns a multimodal world model that predicts future text and image representations and learns to act from imagined model rollouts. Unlike traditional agents that use language only to predict actions, Dynalang acquires rich language understanding by using past language also to predict future language, video, and rewards. In addition to learning from online interaction in an environment, Dynalang can be pretrained on datasets of text, video, or both without actions or rewards. From using language hints in grid worlds to navigating photorealistic scans of homes, Dynalang utilizes diverse types of language to improve task performance, including environment descriptions, game rules, and instructions.

or 

> Summary:
The paper presents Dynalang, an agent that integrates language understanding with future prediction as a self-supervised learning objective. Dynalang predicts future text and image representations and learns to act from imagined model rollouts, using language to enhance task performance in various settings. The agent leverages diverse types of language to improve its understanding and action predictions in different environments.

[GPT-4V(ision) system card](https://openai.com/research/gpt-4v-system-card)

> Summary:
The abstract discusses the introduction of GPT-4 with Vision (GPT-4V), which allows users to command GPT-4 to analyze image inputs. This enhancement is part of an effort in the field of artificial intelligence to integrate various modalities like image inputs into large language models (LLMs), extending their capabilities and applications. The text outlines the potential for multimodal LLMs to tackle new tasks and offer unique experiences. The authors specifically delve into the safety aspects of GPT-4V, examining the evaluations, preparations, and mitigation efforts tailored for image inputs, building upon the existing safety work done for GPT-4.

[Training diffusion models with reinforcement learning](https://arxiv.org/pdf/2305.13301)
Black, Kevin, et al. "Training diffusion models with reinforcement learning." arXiv preprint arXiv:2305.13301 (2023).

> Abstract: Diffusion models are a class of flexible generative models trained with an approximation to the log-likelihood objective. However, most use cases of diffusion models are not concerned with likelihoods, but instead with downstream objectives such as human-perceived image quality or drug effectiveness. In this paper, we investigate reinforcement learning methods for directly optimizing diffusion models for such objectives. We describe how posing denoising as a multi-step decision-making problem enables a class of policy gradient algorithms, which we refer to as denoising diffusion policy optimization (DDPO), that are more effective than alternative reward-weighted likelihood approaches. Empirically, DDPO is able to adapt text-to-image diffusion models to objectives that are difficult to express via prompting, such as image compressibility, and those derived from human feedback, such as aesthetic quality. Finally, we show that DDPO can improve prompt-image alignment using feedback from a vision-language model without the need for additional data collection or human annotation.

or

> Summary:
This research investigates the use of reinforcement learning for optimizing diffusion models based on downstream objectives such as image quality or drug effectiveness. It introduces denoising diffusion policy optimization (DDPO), a more effective policy gradient algorithm compared to alternative reward-weighted likelihood approaches. DDPO adapts text-to-image diffusion models to objectives that are challenging to express via prompting, improving prompt-image alignment using feedback from a vision-language model without additional data or human annotation.

## Offline / Off-policy RL
I am less certain that these papers are worthy and interesting, but wanted to include some advances in this space due to the survey feedback. I would love to hear what you think of these and if you have different ideas

### [Behavior Proximal Policy Optimization](https://arxiv.org/pdf/2302.11312.pdf)
Applying offline learning to PPO

### [Harnessing Mixed Offline Reinforcement Learning Datasets via Trajectory Weighting](https://arxiv.org/pdf/2306.13085.pdf)
By reweighting an offline trajectory dataset and retraining, they achieve better results in learning from the dataset in a way that's agnostic to the offline RL training algorithm used

[A Closer Look at Offline RL Agents](https://proceedings.neurips.cc/paper_files/paper/2022/file/3908cadfcc99db12001eafb1207353e9-Paper-Conference.pdf)
Fu, Yuwei, Di Wu, and Benoit Boulet. "A closer look at offline rl agents." Advances in Neural Information Processing Systems 35 (2022): 8591-8604.

[Pre-Training for Robots: Offline RL Enables Learning New Tasks in a Handful of Trials](https://arxiv.org/pdf/2210.05178)
Kumar, Aviral, et al. "Pre-training for robots: Offline rl enables learning new tasks from a handful of trials." arXiv preprint arXiv:2210.05178 (2022).

[Cal-QL: Calibrated Offline RL Pre-Training for Efficient Online Fine-Tuning](https://arxiv.org/pdf/2303.05479)
Nakamoto, Mitsuhiko, et al. "Cal-ql: Calibrated offline rl pre-training for efficient online fine-tuning." arXiv preprint arXiv:2303.05479 (2023).

[A Survey on Offline Reinforcement Learning: Taxonomy, Review, and Open Problems](https://ieeexplore.ieee.org/iel7/5962385/6104215/10078377.pdf)
Prudencio, Rafael Figueiredo, Marcos ROA Maximo, and Esther Luna Colombini. "A survey on offline reinforcement learning: Taxonomy, review, and open problems." IEEE Transactions on Neural Networks and Learning Systems (2023).


## Imitation Learning

[Hierarchical Model-Based Imitation Learning for Planning in Autonomous Driving](https://arxiv.org/pdf/2210.09539)
Bronstein, Eli, et al. from Waymo "Hierarchical model-based imitation learning for planning in autonomous driving." 2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2022.

[A Survey on Imitation Learning Techniques for End-to-End Autonomous Vehicles](https://ieeexplore.ieee.org/abstract/document/9700770)
Le Mero, Luc, et al. "A survey on imitation learning techniques for end-to-end autonomous vehicles." IEEE Transactions on Intelligent Transportation Systems 23.9 (2022): 14128-14147.


## MARL and Meta-RL
These were the ones you provided, so I haven't had a chance to look into them too deeply but I'm planning to include them

### [MAESTRO: Open-Ended Environment Design for Multi-Agent Reinforcement Learning](https://arxiv.org/pdf/2303.03376.pdf)

### [Human-Timescale Adaptation in an Open-Ended Task Space](https://arxiv.org/pdf/2301.07608.pdf)


[Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](http://proceedings.mlr.press/v70/finn17a/finn17a.pdf)
Finn, Chelsea, Pieter Abbeel, and Sergey Levine. "Model-agnostic meta-learning for fast adaptation of deep networks." International conference on machine learning. PMLR, 2017.

[Stabilizing Unsupervised Environment Design with a Learned Adversary](https://arxiv.org/abs/2308.10797)
Mediratta, Ishita, et al. "Stabilizing Unsupervised Environment Design with a Learned Adversary." arXiv preprint arXiv:2308.10797 (2023).

[Cal-QL: Calibrated Offline RL Pre-Training for Efficient Online Fine-Tuning](https://arxiv.org/pdf/2303.05479)
Nakamoto, Mitsuhiko, et al. "Cal-ql: Calibrated offline rl pre-training for efficient online fine-tuning." arXiv preprint arXiv:2303.05479 (2023).

[Deep Reinforcement Learning with Plasticity Injection](https://arxiv.org/pdf/2305.15555)
Nikishin, Evgenii, et al. "Deep Reinforcement Learning with Plasticity Injection." arXiv preprint arXiv:2305.15555 (2023).


[GNM: A General Navigation Model to Drive Any Robot](https://arxiv.org/pdf/2210.03370.pdf)
Shah, Dhruv, et al. "Gnm: A general navigation model to drive any robot." 2023 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2023.


[META-REINFORCEMENT LEARNING: ALGORITHMS AND APPLICATIONS](https://stacks.stanford.edu/file/druid:zf342ty7446/final_thesis-augmented.pdf)

[A Survey of Meta-Reinforcement Learning](https://arxiv.org/pdf/2301.08028)
Beck, Jacob, et al. "A survey of meta-reinforcement learning." arXiv preprint arXiv:2301.08028 (2023).




## Hierarchical RL

[Hierarchical Planning Through Goal-Conditioned Offline Reinforcement Learning](https://arxiv.org/pdf/2205.11790)
Li, Jinning, et al. "Hierarchical planning through goal-conditioned offline reinforcement learning." IEEE Robotics and Automation Letters 7.4 (2022): 10216-10223.

[Multi-Stage Cable Routing through Hierarchical Imitation Learning](https://arxiv.org/abs/2307.08927)
Luo, Jianlan, et al. "Multi-Stage Cable Routing through Hierarchical Imitation Learning." arXiv preprint arXiv:2307.08927 (2023).

[Hierarchical Reinforcement Learning: A Survey and Open Research Challenges](https://www.mdpi.com/2504-4990/4/1/9)
Hutsebaut-Buysse, Matthias, Kevin Mets, and Steven Latré. "Hierarchical reinforcement learning: A survey and open research challenges." Machine Learning and Knowledge Extraction 4.1 (2022): 172-221.
