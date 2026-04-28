import os
os.environ['MUJOCO_GL'] = os.getenv("MUJOCO_GL", 'egl')
os.environ['LAZY_LEGACY_OP'] = '0'
os.environ['TORCHDYNAMO_INLINE_INBUILT_NN_MODULES'] = "1"
os.environ['TORCH_LOGS'] = "+recompiles"
import warnings
warnings.filterwarnings('ignore')
import torch

import hydra
from termcolor import colored

from common.parser import parse_cfg
from common.seed import set_seed
from common.buffer import Buffer
from envs import make_env
from tdmpc2 import TDMPC2
from trainer.offline_trainer import OfflineTrainer
from trainer.online_trainer import OnlineTrainer
from common.logger import Logger

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')


@hydra.main(config_name='config', config_path='.')
def train(cfg: dict):
	"""
	Script for training single-task / multi-task TD-MPC2 agents.

	Most relevant args:
		`task`: task name (or mt30/mt80 for multi-task training)
		`model_size`: model size, must be one of `[1, 5, 19, 48, 317]` (default: 5)
		`steps`: number of training/environment steps (default: 10M)
		`seed`: random seed (default: 1)

	See config.yaml for a full list of args.

	Example usage:
	```
		$ python train.py task=mt80 model_size=48
		$ python train.py task=mt30 model_size=317
		$ python train.py task=dog-run steps=7000000
	```
	"""
	assert torch.cuda.is_available()
	assert cfg.steps > 0, 'Must train for at least 1 step.'
	cfg = parse_cfg(cfg)
	set_seed(cfg.seed)
	print(colored('Work dir:', 'yellow', attrs=['bold']), cfg.work_dir)

	trainer_cls = OfflineTrainer if cfg.multitask else OnlineTrainer
	env = make_env(cfg)
	agent = TDMPC2(cfg)
	init_from = cfg.get('init_from', None)
	sd_supervised_init_from = cfg.get('sd_supervised_init_from', None)
	resume_from = cfg.get('resume_from', None)
	if init_from and resume_from:
		raise ValueError('init_from and resume_from are mutually exclusive.')
	if sd_supervised_init_from and init_from:
		raise ValueError('sd_supervised_init_from and init_from are mutually exclusive.')
	if init_from:
		import os as _os
		assert _os.path.exists(init_from), f'init_from checkpoint not found: {init_from}'
		print(colored(f'Loading model weights from: {init_from}', 'cyan', attrs=['bold']))
		agent.load(init_from)
	if sd_supervised_init_from:
		import os as _os
		assert _os.path.exists(sd_supervised_init_from), f'sd_supervised_init_from checkpoint not found: {sd_supervised_init_from}'
		assert cfg.world_model_kind == 'sd_hopper', 'sd_supervised_init_from only applies to world_model_kind=sd_hopper.'
		assert hasattr(agent.model, 'load_supervised_dynamics'), 'Current world model does not support supervised SD init.'
		print(colored(f'Loading supervised SD dynamics from: {sd_supervised_init_from}', 'cyan', attrs=['bold']))
		load_info = agent.model.load_supervised_dynamics(sd_supervised_init_from)
		print(colored(
			f"Loaded {load_info['loaded_keys']} SD dynamics tensors "
			f"(missing={len(load_info['missing_keys'])}, unexpected={len(load_info['unexpected_keys'])})",
			'cyan',
		))
	trainer = trainer_cls(
		cfg=cfg,
		env=env,
		agent=agent,
		buffer=Buffer(cfg),
		logger=Logger(cfg),
	)
	if resume_from:
		import os as _os
		assert _os.path.exists(resume_from), f'resume_from checkpoint not found: {resume_from}'
		print(colored(f'Resuming full state from: {resume_from}', 'cyan', attrs=['bold']))
		trainer.load_full_state(resume_from)
	trainer.train()
	print('\nTraining completed successfully')


if __name__ == '__main__':
	train()
