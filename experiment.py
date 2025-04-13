from train_agent import *
from play_with_dqn_agent import *


def train_dqn_save_model_stats(args_dict):
    # First, we train a model without any unlearning method.
    clean_train_scores, clean_train_writer = train_dqn(n_episodes=args['episodes'], max_t=args['max-steps'], eps_start=args['eps-start'], 
            eps_end=['eps-end'], eps_decay=args['eps-decay'], checkpoint_freq=args['checkpoint-freq'], learning_rate=['learning-rate'], 
            alpha=args['alpha'], beta=args['beta'], beta_frames=args['beta-frames'], buffer_size=args['buffer-size'], 
            batch_size=args['batch-size'], gamma=args['gamma'], model_dir=args['model-dir'], 
            log_dir=args['log-dir'], decay_type=args['decay-type'], decay_percent=args['decay-percent'], unlearning_type="none")

    # Then, we do an analysis with the 
    #python play_with_dqn_agent.py --model best_model.pth --mode compare --num-games 10000
    #compare_with_optimal(args.model, args.num_games)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run an Unlearning Experiment')
    parser.add_argument('--episodes', type=int, default=110000, help='Number of episodes to train')
    parser.add_argument('--max-steps', type=int, default=100, help='Maximum steps per episode')
    parser.add_argument('--eps-start', type=float, default=1.0, help='Starting epsilon value')
    parser.add_argument('--eps-end', type=float, default=0.01, help='Minimum epsilon value')
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

    args_dict = vars(args)


