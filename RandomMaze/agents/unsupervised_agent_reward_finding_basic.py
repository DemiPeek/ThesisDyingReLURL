
from networks import EncoderDMC, DQNmodel_shallow, EncoderDMC_lowdim
from utils import to_numpy, count_dying_relu, count_time_dead
from replaybuffer import ReplayBuffer
import torch
import torch.nn.functional as F
import numpy as np
import random as r
import wandb


class Agent_Reward_Finding_Basic:

    def __init__(self, env, eval_env, args=None):

        if torch.cuda.is_available() and args.device == 'gpu':
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        # Environment variables
        self.name = 'fourmaze_3states'
        self.env = env
        self.eval_env = eval_env
        self.maze_size = env._size_maze
        self.higher_dim_obs = env._higher_dim_obs

        # Algorithmic variables
        self.latent_dim = args.latent_dim
        self.batch_size = args.batch_size
        self.eps = args.eps_start
        self.eps_end = args.eps
        self.tau = args.tau
        self.gamma = args.gamma
        self.lr = args.lr_encoder
        self.lr_dqn = args.lr_dqn
        self.activation = args.activation
        self.experiment = args.experiment
        self.reincarnate_type = args.reincarnate_type
        self.dead_timecount = torch.tensor(np.zeros(args.latent_dim, dtype="float32")).to(self.device)
        self.relu_bound = args.relu_bound
        self.reward_scale = args.reward_scale
        self.optimizer = args.optimizer

        self.onehot = True
        self.action_dim = 4
        self.iterations = 0

        if self.higher_dim_obs:
            # Convolutional Encoder (Slower, for pixel images)
            self.encoder = EncoderDMC(latent_dim=self.latent_dim, experiment=self.experiment, maze_size=self.maze_size, activation=self.activation).to(self.device)
        else:
            # MLP Encoder (Faster, not for pixel images)
            self.encoder = EncoderDMC_lowdim(latent_dim=self.latent_dim, maze_size=self.maze_size, activation=self.activation).to(self.device)

        # DQN network & Target network
        self.dqn = DQNmodel_shallow(input_dim=self.latent_dim, depth=1).to(self.device)
        self.target_dqn = DQNmodel_shallow(input_dim=self.latent_dim, depth=1).to(self.device)
        self.target_dqn.load_state_dict(self.dqn.state_dict())

        # Optimizer encoder
        if self.optimizer == "adam":
            self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=self.lr)
        elif self.optimizer == "sgd":
            self.encoder_optimizer = torch.optim.SGD(self.encoder.parameters(), lr=self.lr)
        elif self.optimizer == "sgd_momentum":
            self.encoder_optimizer = torch.optim.SGD(self.encoder.parameters(), lr=self.lr, momentum=0.9)

        # Optimizer DQN
        self.dqn_optimizer = torch.optim.Adam(self.dqn.parameters(), lr=self.lr_dqn)

        # Replay Buffer
        self.buffer = ReplayBuffer(np.expand_dims(self.env.observe()[0], axis=0).shape, env.action_space.shape[0], int(args.fill_buffer)+int(args.iterations)+50000, self.device)

        self.best_reward = -999
        self.average_reward = 0
        self.output = {}

        self.initial_weights = {name: param.clone().detach() for name, param in self.encoder.named_parameters()}

    def mlp_learn(self):

        STATE, ACTION, REWARD, NEXT_STATE, DONE = self.buffer.sample(self.batch_size)
        REWARD = REWARD * self.reward_scale
        # Remove gradients
        for param in self.encoder.parameters():
            param.grad = None
        for param in self.dqn.parameters():
            param.grad = None

        # One-hot action encodings
        if self.onehot:
            ACTION = F.one_hot(ACTION.squeeze(1).long(), num_classes=self.action_dim)

        # Current latents (Z_t)
        full_latent, full_features, without_activation = self.encoder(STATE, self.dead_timecount)
        # Next latents (Z_t+1)
        next_latent, next_features, next_without_activation = self.encoder(NEXT_STATE, self.dead_timecount)

        Q = self.dqn(full_latent)
        next_Q = self.dqn(next_latent)
        target_Q = self.target_dqn(next_latent)
        q_loss = self.compute_DDQN_Loss(Q, next_Q, target_Q, ACTION, REWARD, DONE)

        loss = q_loss
        if self.experiment == 'bounded_loss':
            infinity_norm = torch.norm(full_latent, float('inf'), dim=-1) - self.relu_bound
            zero = torch.zeros_like(infinity_norm)
            max_value = torch.max(infinity_norm, zero).mean()
            loss += max_value

        if self.activation == 'relu':
            if self.iterations % 100 == 0:
                weight, dead = count_dying_relu(full_latent)
                frac_weighteddead = weight/self.latent_dim
                frac_fullydead = dead/self.latent_dim
                print('The fraction of dead ReLU is', frac_fullydead)
                print('The fraction of weighted dead ReLU is', frac_weighteddead)

                wandb.log({"fully_dead_relu": dead,
                           "weighted_dead_relu": weight,
                           "fraction_weighted_dead_relu": frac_weighteddead,
                           "fraction_fully_dead_relu": frac_fullydead,
                           'global_iterations': self.iterations})

            if self.experiment == 'reincarnate':
                # function to update dead_timecount
                self.dead_timecount, lrelu_count = count_time_dead(self.dead_timecount, self.reincarnate_type, full_latent)
                frac_lrelu = lrelu_count.sum()/self.latent_dim
                nr_lrelu = lrelu_count.sum()
                wandb.log({"fraction_leaky_relu": frac_lrelu,
                           "nr_leaky_relu": nr_lrelu,
                           'global_iterations': self.iterations})

        if self.iterations % 100 == 0:
            wandb.log({
                    "q_loss": q_loss,
                    'global_iterations': self.iterations
                       })

        # Backprop the loss
        loss.backward()

        self.encoder_optimizer.step()
        self.dqn_optimizer.step()

        if self.iterations % 100 == 0:
            # Metrics weights of layer before activation
            weights = self.encoder.mlp_without_activation.weight
            weights_array = weights.cpu().detach().numpy()
            mean_weights = np.mean(weights_array)
            std_weights = np.std(weights_array)
            nr_neg_weights = np.sum(weights_array <= 0)
            ratio_neg_weights = nr_neg_weights/np.size(weights_array)

            # Metrics of layer before activation
            without_activation_array = without_activation.cpu().detach().numpy()
            mean_without_activation = np.mean(without_activation_array)
            std_without_activation = np.std(without_activation_array)
            nr_neg_without_activation = np.sum(without_activation_array <= 0)
            ratio_neg_without_activation = nr_neg_without_activation/np.size(without_activation_array)

            wandb.log({'mean_weights': mean_weights,
                       'std_weights': std_weights,
                       'ratio_neg_weights': ratio_neg_weights,
                       'mean_layer_without_activation': mean_without_activation,
                       'std_layer_without_activation': std_without_activation,
                       'ratio_layer_neg_without_activation': ratio_neg_without_activation,
                       'global_iterations': self.iterations
                       })

        # target DQN network update
        for target_param, param in zip(self.target_dqn.parameters(), self.dqn.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

        # Print the losses and predictions every 500 iterations
        if self.iterations % 500 == 0:
            print("Iterations", self.iterations)

        self.iterations += 1

    def get_action(self, latent):

        if np.random.rand() < self.eps:
            return self.env.actions[r.randrange(4)]
        else:
            with torch.no_grad():
                q_vals = self.dqn(latent)
            action = np.argmax(to_numpy(q_vals))
            return action

    def run_agent(self):

        done = False
        state = self.env.observe()
        latent, _, _1 = self.encoder(torch.as_tensor(state).unsqueeze(0).float().to(self.device), self.dead_timecount)
        action = self.get_action(latent)
        reward = self.env.step(action)
        next_state = self.env.observe()

        if self.env.inTerminalState():
            self.env.reset(1)
            done = True
        self.buffer.add(state, action, reward, next_state, done)

        if self.iterations % 100 == 0:
            # self.average_reward = self.output['average_reward']
            if self.iterations == 0:
                self.evaluate(eval_episodes=100)
            else:
                self.evaluate()

        self.mlp_learn()
        self.eps = max(self.eps_end, self.eps - 0.8/50000)

    def evaluate(self, eval_episodes=100, give_value=False):

        self.eval_env.reset(1)
        average_reward = []
        Average_reward = 0

        for i in range(eval_episodes):
            reward = []
            done=False
            while not done:

                state = self.eval_env.observe()
                latent, _, _1 = self.encoder(torch.as_tensor(state).unsqueeze(0).float().to(self.device), self.dead_timecount)

                action = self.get_action(latent)
                reward_t = self.eval_env.step(action, dont_take_reward=False)
                reward.append(reward_t)

                if self.eval_env.inTerminalState():
                    self.eval_env.reset(1)
                    done = True
                    reward = sum(reward)
                    average_reward.append(reward)

                ## Uncomment this if you want to see the agent in action
                # if self.iterations >= 150000:
                #     img = plt.imshow(state[0], cmap='gray')
                #     plt.pause(0.5)
                #     plt.draw()

        Average_reward += sum(average_reward)/len(average_reward)
        wandb.log({'average_reward': Average_reward,
                  'global_iterations': self.iterations})

        print('The AVERAGE REWARD is:', Average_reward)
        if Average_reward >= self.best_reward:
            self.best_reward = Average_reward
            wandb.log({'Best reward': self.best_reward,
                       'global_iterations': self.iterations})

        if give_value:
            return Average_reward

    def compute_DDQN_Loss(self, Q, next_Q, target_Q, actions, rewards, dones):

        # Change actions to long format for the gather function
        actions = actions.long()
        actions = torch.argmax(actions, dim=1).unsqueeze(1)
        # Compute the Q-value estimates corresponding to the actions in the batch
        target_actions = torch.argmax(next_Q, dim=1).unsqueeze(1)
        # We use the target Q-network and fill in the actions from the 'original' Q-network
        target_Q_value = target_Q.gather(1, target_actions)
        # Current timestep Q-values with the actions from the minibatch
        Q = Q.gather(1, actions)
        Q_target = rewards + (1 - dones.int()) * self.gamma * target_Q_value
        loss = F.mse_loss(Q, Q_target.detach())

        return loss
