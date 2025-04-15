from train_agent import train_dqn
from play_with_dqn_agent import compare_with_optimal
from poker_env import TARGETS
import argparse


def run_experiment(args, best_model_dir = 'models/best_normal'):
    # First, we train a model without any unlearning method.
    if args['train_normal_model']:
        clean_train_scores, clean_train_writer, clean_model_dir = train_dqn(n_episodes=args['episodes'], max_t=args['max_steps'], eps_start=args['eps_start'], 
                eps_end=args['eps_end'], eps_decay=args['eps_decay'], checkpoint_freq=args['checkpoint_freq'], learning_rate=args['learning_rate'], 
                alpha=args['alpha'], beta=args['beta'], beta_frames=args['beta_frames'], buffer_size=args['buffer_size'], 
                batch_size=args['batch_size'], gamma=args['gamma'], model_dir=args['model_dir'], 
                log_dir=args['log_dir'], decay_type=args['decay_type'], decay_percent=args['decay_percent'], unlearning_type="none")
    else:
        clean_model_dir = best_model_dir

    # Then, we do an analysis on that one
    compare_with_optimal(f'{clean_model_dir}/best_model.pth', num_games = args['eval_episodes'], eval_dir_name=f"models/{args['save_name']}")

    # Then, we 
    unlearned_train_scores, unlearned_train_writer, unlearned_model_dir = train_dqn(n_episodes=args['episodes'], max_t=args['max_steps'], eps_start=args['eps_start'], 
            eps_end=args['eps_end'], eps_decay=args['eps_decay'], checkpoint_freq=args['checkpoint_freq'], learning_rate=args['learning_rate'], 
            alpha=args['alpha'], beta=args['beta'], beta_frames=args['beta_frames'], buffer_size=args['buffer_size'], 
            batch_size=args['batch_size'], gamma=args['gamma'], model_dir=args['model_dir'], 
            log_dir=args['log_dir'], decay_type=args['decay_type'], decay_percent=args['decay_percent'], unlearning_type=args['unlearning_type'], model_path=f'{clean_model_dir}/best_model.pth', save_name = args['save_name'])

    # Then, we do an analysis on that one
    compare_with_optimal(f'{unlearned_model_dir}/best_model.pth', f'{clean_model_dir}/best_model.pth', args['eval_episodes'], eval_dir_name=f"models/{args['save_name']}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run an Unlearning Experiment')
    parser.add_argument('--episodes', type=int, default=100000, help='Number of episodes to train')
    parser.add_argument('--max-steps', type=int, default=100, help='Maximum steps per episode')
    parser.add_argument('--eps-start', type=float, default=0.15, help='Starting epsilon value')
    parser.add_argument('--eps-end', type=float, default=0.0001, help='Minimum epsilon value')
    parser.add_argument('--eps-decay', type=float, default=0.995, help='Epsilon decay factor (for exponential decay)')
    parser.add_argument('--checkpoint-freq', type=int, default=10_000_000, help='Checkpoint frequency (episodes)')
    parser.add_argument('--model-dir', type=str, default='models', help='Directory to save models')
    parser.add_argument('--log-dir', type=str, default='runs/video_poker', help='Directory to save TensorBoard logs')
    parser.add_argument('--decay-type', type=str, choices=['exponential', 'linear'], default='linear', 
                       help='Type of epsilon decay schedule')
    parser.add_argument('--decay-percent', type=float, default=80, 
                       help='Percentage of episodes over which to decay epsilon (for linear decay)')
    parser.add_argument('--buffer-size', type=int, default=40_000, help='Size of replay buffer')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--alpha', type=float, default=0.6, help='Alpha parameter for Prioritized Experience Replay')
    parser.add_argument('--beta', type=float, default=0.4, help='Beta parameter for Prioritized Experience Replay')
    parser.add_argument('--beta-frames', type=int, default=100_000, help='Number of frames to decay beta')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--gamma', type=float, default=1.0, help='Discount factor')
    parser.add_argument('--eval-episodes', type = int, default=20000)
    parser.add_argument('--unlearning-type', type=str, choices=['decremental', 'env-poisoning'], default='decremental', help='Type of unlearning type to test')
    parser.add_argument('--train-normal-model', type=bool, default = False)
    parser.add_argument('--from-model-dir', type=str, default = 'models/best_normal')
    parser.add_argument('--save-name', type=str)

    args_dict = vars(parser.parse_args())
    run_experiment(args_dict, best_model_dir=args_dict['from_model_dir'])


