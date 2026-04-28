import os
os.environ['MUJOCO_GL'] = os.getenv("MUJOCO_GL", 'glfw')
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
	resume_from = cfg.get('resume_from', None)
	if init_from and resume_from:
		raise ValueError('init_from and resume_from are mutually exclusive.')
	if init_from:
		import os as _os
		assert _os.path.exists(init_from), f'init_from checkpoint not found: {init_from}'
		print(colored(f'Loading model weights from: {init_from}', 'cyan', attrs=['bold']))
		agent.load(init_from)
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
