## Algorithm
To solve the Tennis environment task, I tried to use multiple agnet ddpg model (MADDPG) and single agent DDPG with actor-critic mechanism.

DDPG is an algorithm which learns Q-function and a policy at the same time. Q-function is learned 
using off-policy trials and later used to learn the optimal policy. 

It consists of two models - actor and critic. Actor is a policy network that takes the state as 
input and outputs the exact continuous action. Actions are learned deterministicly, which means
that actor outputs action directly instead of a probability distribution over actions.

Critic is a Q-value network that takes in state and action as input and outputs the Q-value. 
Critic here behaves as a "teacher", evaluating the actions that were provided by the actor.

## Model Architecture
For actor, I used two hidden units with 400 and 300 hidden units respectively. For each hidden layer
ReLU activation function was used. Actor outputs two values corresponding to agent actions. For 
output layer we used tanh activation function in order to normalize the output between -1 an 1.

Critic network received concatenated states of both agents with actions from the actor output. We used 
2 hidden layers with 400 and 300 hidden units. ReLU activation was used for each hidden layer. 
and output of 1 unit. 


###Single agent DDPG
Final model is trained using single agent DDPG algorithm that achieved desired reward df of 0.5 as average over 100 episode. 
Full training could be seen on following chart. Results are an average over 100 episodes, as stated in task description. 

![ddpg](results/ddpg/result_plot.png)

I concatenated states of both agents and feed it to the actor. Final action values were used for 
both players, so they are moving in the same (or opposite to be precise) direction.   

DDPG was trained on a single Agent and trained on CPU only.I used convolutional network with two 
hidden layers of 400 and 300 units for actor and critic, learning rate of actor was set to 1e-3 and 
critic to 5e-4, I used ReplayBuffer to pull sample examples to train network and soft-updated 
that updated target network by some changes in local network. I updated the network after every step I took.

### Multi agent DDPG
Unfortunately, I was unable to make MADDPG to work. After many trials, extensive hyperparameter 
tuning and feature addition. the algorithm still didnt converge to desired reward. best model using MADDPG 
could be seen on following chart:
![maddpg](results/maddpg/result_plot.png)
As you can see, the results were nearly as good as with single agent ddpg. This result was achieved 
after many tens of thousands of episodes, pretrained model and extensive hyperparameter tuning. I 
applied OUNOISE with decreasing noise parameter, prioritized replay buffer with rewards only higher 
than x. I also tried tu tune both actor and critic architectures, both learning rates, soft update 
parameter and discount rate. Unfortunatelly nothing helped the algorithm to get better results. I
believe there is some bug in our code that I just cannot see.


## Navigation
To run the training, edit the hyperparameters in `config.py` and run the script either for single agent DDPG
```python
python main_singleagent.py
```
or multi agnet ddpg
```python
python main.py
```
DDPG algorithm is written in `ddpg.py` and multiagent network with ddpg algorithm is in `maddpg_tennis.py`.

Final model is saved in `results/maddpg/episode-4999.pt`. In this folder we can find trained model, hyperparameters used with trained model and final graph depicting the training.
Model is trained from single-agent ddpg algorithm, as I unfortunatelly couldnt find a way to make multiple agent DDPG to work. 

All config values are set in `config.py` module.

To simply evaluate the results and see tennis agent perform, run
```python 
python evaluation.py --n-episode 10
```
To see the agent perform on 10 episodes.

Repo also contains `workspace_udacite` that served as a baseline for our maddpg algorithm, but is not use in any way in final model.

Final algorithm is further described in REPORT.md
## Hyperparameters
All hyperparameter values are stored in `results/ddpg/config_params.json`

##Trials
We performed extensive hyperparameter tuning with MADDPG algorithm. After tuning learning rate of 
both actor and critic, network architecture and number of layers, tau and gamma parameter, I 
decided to use only singlie agent DDPG that worked flawlessly on first try. I hope to get back
to this task later in the future and solve it with MADDPG as well.


## Possible improvements
Possible improvement could be using different algorithms than ddpg or maddpg. As maddpg turned out
ineffective for us, we could try for example TD3 algorithm

Another possible extension could be to continue training with our pretrained final model with lower 
noise or lower learning rates for actor and critic. This could lead us to better results than achieved 0.7.