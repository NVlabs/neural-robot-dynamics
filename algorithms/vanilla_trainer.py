# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys, os
base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
sys.path.append(base_dir)

import time
import shutil
from typing import Optional

import warp as wp
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.clip_grad import clip_grad_norm_
import yaml
import numpy as np
from tqdm import tqdm

from envs.neural_environment import NeuralEnvironment
from models.models import ModelMixedInput
from utils.datasets import BatchTransitionDataset, collate_fn_BatchTransitionDataset
from utils.evaluator import NeuralSimEvaluator
from utils.python_utils import (
    set_random_seed, 
    print_info, print_ok, print_white, print_warning,
    format_dict
)
from utils.torch_utils import num_params_torch_model, grad_norm
from utils.running_mean_std import RunningMeanStd
from utils.time_report import TimeReport, TimeProfiler
from utils.logger import Logger

class VanillaTrainer:
    def __init__(
        self, 
        neural_env: NeuralEnvironment, 
        cfg: dict, 
        model_checkpoint_path: Optional[str] = None, 
        device = 'cuda:0'
    ):
    
        algo_cfg = cfg['algorithm']
        cli_cfg = cfg['cli']

        self.seed = algo_cfg.get('seed', 0)
        self.device = device

        set_random_seed(self.seed)

        self.rng = np.random.default_rng(seed = self.seed)

        self.neural_env = neural_env
        self.neural_integrator = neural_env.integrator_neural

        # check if gravity_dir_body is included in input if using body frame
        if cfg['env']['neural_integrator_cfg']['states_frame'] == 'body':
            if 'gravity_dir' not in cfg['inputs']['low_dim']:
                cfg['inputs']['low_dim'].append('gravity_dir')
                print_warning("gravity_dir not included in low_dim inputs, "
                              "added it automatically.")

        # create neural sim model
        if model_checkpoint_path is None:
            input_sample = self.neural_integrator.get_neural_model_inputs()
            self.neural_model = ModelMixedInput(
                input_sample = input_sample,
                output_dim = self.neural_integrator.prediction_dim,
                input_cfg = cfg['inputs'],
                network_cfg = cfg['network'],
                device = self.device
            )
        else:
            checkpoint = torch.load(model_checkpoint_path, map_location=self.device)
            self.neural_model = checkpoint[0]
            self.neural_model.to(self.device)
        
        print('Model = \n', self.neural_model)
        print('# Model Parameters = ', num_params_torch_model(self.neural_model))

        self.neural_integrator.set_neural_model(self.neural_model)

        """ General parameters """
        
        # batch_size has to be loaded before get_datasets call since
        # it might get reset in get_datasets
        self.batch_size = int(algo_cfg['batch_size'])
        self.num_valid_batches = int(algo_cfg.get('num_valid_batches', 50))

        # load datasets
        self.dataset_max_capacity = algo_cfg['dataset'].get('max_capacity', 100000000)
        self.num_data_workers = algo_cfg['dataset'].get('num_data_workers', 4)
        train_dataset_path = algo_cfg['dataset'].get('train_dataset_path', None)
        valid_datasets_cfg = algo_cfg['dataset'].get('valid_datasets', None)
        
        self.train_dataset = None
        self.valid_datasets = {}
        self.collate_fn = None
        
        self.batch_size = int(algo_cfg['batch_size'])
        
        self.get_datasets(train_dataset_path, valid_datasets_cfg)

        """ Parameters only used for training """
        if cli_cfg['train']:
            # load training general parameters
            self.num_epochs = int(algo_cfg['num_epochs'])
            self.num_iters_per_epoch = int(algo_cfg.get('num_iters_per_epoch', -1))

            # load learning rate params
            self.lr_start = float(algo_cfg['optimizer']['lr_start'])
            self.lr_end = float(algo_cfg['optimizer'].get('lr_end', 0.))
            self.lr_schedule = algo_cfg['optimizer']['lr_schedule']
            self.optimizer = torch.optim.Adam(self.neural_model.parameters(), 
                                              lr = self.lr_start)

            # load gradient clipping params
            self.truncate_grad = algo_cfg.get('truncate_grad', False)
            self.grad_norm = algo_cfg.get('grad_norm', 1.0)

            # logging related
            self.log_dir = cli_cfg["logdir"]
            if os.path.exists(self.log_dir) and not cli_cfg["skip_check_log_override"]:
                ans = input(f"Logging Directory {self.log_dir} exist, overwrite? [y/n]")
                if ans == 'y':
                    shutil.rmtree(self.log_dir)
                else:
                    exit
                
            os.makedirs(self.log_dir, exist_ok = True)

            self.model_log_dir = os.path.join(self.log_dir, 'nn')
            os.makedirs(self.model_log_dir, exist_ok = True)
            
            self.summary_log_dir = os.path.join(self.log_dir, 'summaries')
            os.makedirs(self.summary_log_dir, exist_ok = True)

            # save config
            yaml.dump(cfg, open(os.path.join(self.log_dir, 'cfg.yaml'), 'w'))

            # create logger
            self.logger = Logger()
            self.logger.init_tensorboard(self.summary_log_dir)
                
            # other logging params
            self.save_interval = cli_cfg.get("save_interval", 50)
            self.log_interval = cli_cfg.get("log_interval", 1)
            self.eval_interval = cli_cfg.get("eval_interval", 1)

            # do not need to compute dataset statistics if doing finetuning
            if algo_cfg.get("compute_dataset_statistics", True):
                # get dataset mean/std info
                print('Computing dataset statistics...')
                self.compute_dataset_statistics(self.train_dataset)
                print('Finished computing dataset statistics...')
                self.neural_model.set_input_rms(self.dataset_rms)
                self.neural_model.set_output_rms(self.dataset_rms['target'])
            else:
                assert model_checkpoint_path is not None, \
                    "model_checkpoint_path is required to skip computing dataset statistics"
                print_info('Skip computing dataset statistics')
            
            # create logging files for saved best valid model
            for valid_dataset_name in self.valid_datasets.keys():
                fp = open(
                    os.path.join(
                        self.model_log_dir, 
                        f'saved_best_valid_{valid_dataset_name}_model_epochs.txt'
                    ), 'w'
                )
                fp.close()

            fp = open(
                os.path.join(
                    self.model_log_dir, 
                    "saved_best_eval_model_epochs.txt"
                ), 'w'
            )
            fp.close()

        # Create evaluator
        self.eval_mode = algo_cfg['eval'].get('mode', 'sampler')
        self.eval_horizon = algo_cfg['eval'].get("rollout_horizon", 5)
        self.num_eval_rollouts = algo_cfg['eval'].get("num_rollouts", self.neural_env.num_envs)
        self.eval_dataset_path = algo_cfg['eval'].get('dataset_path', None)
        self.eval_passive = algo_cfg['eval'].get('passive', True)
        self.eval_render = cli_cfg['render']

        if self.eval_mode == 'dataset':
            assert self.eval_dataset_path is not None, \
                "If eval_mode is 'dataset', 'eval_dataset_path' must be provided"
            
        self.evaluator = NeuralSimEvaluator(
                            self.neural_env,
                            hdf5_dataset_path = self.eval_dataset_path if self.eval_mode == 'dataset' else None,
                            eval_horizon = self.eval_horizon,
                            device = self.device
                        )
    
    def get_datasets(self, train_dataset_path, valid_datasets_cfg):
        self.train_dataset = BatchTransitionDataset(
                                batch_size = self.batch_size,
                                hdf5_dataset_path = train_dataset_path,
                                max_capacity = self.dataset_max_capacity,
                                device = self.device
                            )

        valid_dataset_names = valid_datasets_cfg.keys()
        for valid_dataset_name in valid_dataset_names:
            self.valid_datasets[valid_dataset_name] = \
                BatchTransitionDataset(
                    batch_size = self.batch_size,
                    hdf5_dataset_path = valid_datasets_cfg[valid_dataset_name],
                    device = self.device
                )

        self.batch_size = 1 # special handling for BatchTransitionDataset

        self.collate_fn = collate_fn_BatchTransitionDataset

    def compute_dataset_statistics(self, dataset):
        # compute the mean and std of the input and output of the dataset
        dataloader = DataLoader(
            dataset = dataset,
            batch_size = max(512, self.batch_size),
            collate_fn = self.collate_fn,
            shuffle = False,
            num_workers = self.num_data_workers,
            drop_last = True
        )
        dataloader_iter = iter(dataloader)
        self.dataset_rms = {}

        for _ in range(len(dataloader)):
            data = next(dataloader_iter)
            data = self.preprocess_data_batch(data)

            for key in data.keys():
                if not (key in self.dataset_rms):
                    self.dataset_rms[key] = RunningMeanStd(
                        shape = data[key].shape[2:],
                        device = self.device
                    )

                self.dataset_rms[key].update(
                    data[key], 
                    batch_dim = True, 
                    time_dim = True
                )

    def get_scheduled_learning_rate(self, iteration, total_iterations):
        if self.lr_schedule == 'constant':
            return self.lr_start
        elif self.lr_schedule == 'linear':
            ratio = iteration / total_iterations
            return self.lr_start * (1.0 - ratio) + self.lr_end * ratio
        elif self.lr_schedule == 'cosine':
            decay_ratio = iteration / total_iterations
            assert 0 <= decay_ratio <= 1
            coeff = 0.5 * (1.0 + np.cos(np.pi * decay_ratio)) # coeff ranges 0..1
            return self.lr_end + coeff * (self.lr_start - self.lr_end)
        else:
            raise NotImplementedError

    @torch.no_grad()
    def preprocess_data_batch(self, data):
        # Move data to target device
        for key in data.keys():
            if type(data[key]) is dict:
                for sub_key in data[key].keys():
                    data[key][sub_key] = data[key][sub_key].to(self.device)
            else:
                data[key] = data[key].to(self.device)
        
        # compute contact masks
        data['contact_masks'] = self.neural_integrator.get_contact_masks(
            data['contact_depths'],
            data['contact_thicknesses']
        )
        
        self.neural_integrator.process_neural_model_inputs(data)

        # calculate prediction target from neural env
        data['target'] = self.neural_integrator.convert_next_states_to_prediction(
                            states = data['states'], 
                            next_states = data['next_states'],
                            dt = self.neural_env.frame_dt
                        )

        return data

    def compute_loss(self, data, train):

        if self.neural_model.is_rnn:
            self.neural_model.init_rnn(self.batch_size)

        prediction_target = data['target']
        prediction = self.neural_model(data)
        
        if self.neural_model.normalize_output:
            loss_weights = 1. / torch.sqrt(self.neural_model.output_rms.var + 1e-5)
        else:
            loss_weights = torch.ones(
                prediction.shape[-1], 
                device = prediction.device
            )
        
        loss = torch.nn.MSELoss()(
            prediction * loss_weights,
            prediction_target * loss_weights
        )

        # Compute reported error statistics defined on next states
        with torch.no_grad():
            predicted_next_states = self.neural_integrator.convert_prediction_to_next_states(
                states = data['states'],
                prediction = prediction
            )
            self.neural_integrator.wrap2PI(predicted_next_states)
            loss_itemized = {}
            for i in range(predicted_next_states.shape[-1]):
                loss_itemized[f'state_{i}'] = ((
                    predicted_next_states[..., i] - data['next_states'][..., i]
                ) ** 2).mean()
            loss_itemized['state_MSE'] = \
                torch.nn.MSELoss()(
                    predicted_next_states,
                    data['next_states']
                )
            loss_itemized['q_error_norm'] = torch.norm(
                predicted_next_states[..., :self.neural_integrator.dof_q_per_env] \
                - data['next_states'][..., :self.neural_integrator.dof_q_per_env],
                dim = -1
            ).mean()
            loss_itemized['qd_error_norm'] = torch.norm(
                predicted_next_states[..., self.neural_integrator.dof_q_per_env:] \
                - data['next_states'][..., self.neural_integrator.dof_q_per_env:],
                dim = -1
            ).mean()

        return loss, loss_itemized
        
    def one_epoch(
        self, 
        train: bool, 
        dataloader, 
        dataloader_iter, 
        num_batches, 
        shuffle = False
    ):
        
        if train:
            self.neural_model.train()
        else:
            self.neural_model.eval()
            
        sum_loss = 0.
        sum_loss_itemized = {}
        if train:
            grad_info = {'grad_norm_before_clip': 0.}
            if self.truncate_grad:
                grad_info['grad_norm_after_clip'] = 0.
        else:
            grad_info = {}

        with torch.set_grad_enabled(train):
            for _ in tqdm(range(num_batches)):
                with TimeProfiler(self.time_report, 'dataloader'):
                    try:
                        data = next(dataloader_iter)
                    except StopIteration:
                        if shuffle:
                            self.train_dataset.shuffle()
                        dataloader_iter = iter(dataloader)
                        data = next(dataloader_iter)

                    data = self.preprocess_data_batch(data)
                
                with TimeProfiler(self.time_report, 'compute_loss'):
                    if train:
                        self.optimizer.zero_grad()

                    loss, loss_itemized = self.compute_loss(data, train)

                with TimeProfiler(self.time_report, 'backward'):
                    if train:
                        loss.backward()

                        # Truncate gradients
                        with torch.no_grad():
                            grad_norm_before_clip = grad_norm(
                                self.neural_model.parameters()
                            )
                            grad_info['grad_norm_before_clip'] += grad_norm_before_clip
                            if self.truncate_grad:
                                clip_grad_norm_(
                                    self.neural_model.parameters(), 
                                    self.grad_norm
                                )
                                grad_norm_after_clip = grad_norm(
                                    self.neural_integrator.neural_model.parameters()
                                ) 
                                grad_info['grad_norm_after_clip'] += grad_norm_after_clip

                        self.optimizer.step()

                with TimeProfiler(self.time_report, 'other'):
                    sum_loss += loss

                    for key in loss_itemized.keys():
                        if key in sum_loss_itemized:
                            sum_loss_itemized[key] += loss_itemized[key]
                        else:
                            sum_loss_itemized[key] = loss_itemized[key]
        
        avg_loss = sum_loss.detach().cpu().item() / num_batches
        avg_loss_itemized = {}
        for key in sum_loss_itemized.keys():
            avg_loss_itemized[key] = sum_loss_itemized[key].cpu().item() / num_batches
        if train:
            grad_info['grad_norm_before_clip'] /= num_batches
            if self.truncate_grad:
                grad_info['grad_norm_after_clip'] /= num_batches

        return avg_loss, avg_loss_itemized, grad_info
            
    def train(self):
        train_loader = DataLoader(
            dataset = self.train_dataset,
            batch_size = self.batch_size,
            collate_fn = self.collate_fn,
            shuffle = True,
            num_workers = self.num_data_workers,
            drop_last = True
        )
        train_loader_iter = iter(train_loader)
        if self.num_iters_per_epoch == -1:
            self.num_train_batches = len(train_loader)
        else:
            self.num_train_batches = self.num_iters_per_epoch

        valid_loaders = {}
        valid_loader_iters = {}
        best_valid_losses = {}
        for valid_dataset_name in self.valid_datasets.keys():
            valid_loaders[valid_dataset_name] = DataLoader(
                dataset = self.valid_datasets[valid_dataset_name],
                batch_size = self.batch_size,
                collate_fn = self.collate_fn,
                shuffle = True,
                num_workers = self.num_data_workers,
                drop_last = True
            )
            valid_loader_iters[valid_dataset_name] = iter(valid_loaders[valid_dataset_name])
            best_valid_losses[valid_dataset_name] = np.inf
            
        self.best_eval_error = np.inf

        self.time_report = TimeReport(cuda_synchronize = False)
        self.time_report.add_timers(
            ['epoch', 'other', 'dataloader', 
             'compute_loss', 'backward', 'eval']
        )

        for epoch in range(self.num_epochs):
            self.time_report.reset_timer()
            
            with TimeProfiler(self.time_report, 'epoch'):
                # Learning rate schedule
                self.lr = self.get_scheduled_learning_rate(epoch, self.num_epochs)

                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lr

                self.logger.init_epoch(epoch)
                
                # Train
                if epoch > 0:
                    avg_train_loss, avg_train_loss_itemized, grad_info = \
                        self.one_epoch(
                            train = True, 
                            dataloader = train_loader,
                            dataloader_iter = train_loader_iter, 
                            num_batches = self.num_train_batches,
                            shuffle = True
                        )

                # Valid
                avg_valid_losses, avg_valid_losses_itemized = {}, {}
                for valid_dataset_name in self.valid_datasets.keys():
                    avg_valid_losses[valid_dataset_name], avg_valid_losses_itemized[valid_dataset_name], _ = \
                        self.one_epoch(
                            train = False, 
                            dataloader = valid_loaders[valid_dataset_name],
                            dataloader_iter = valid_loader_iters[valid_dataset_name], 
                            num_batches = min(
                                self.num_valid_batches, 
                                len(valid_loaders[valid_dataset_name])
                            ),
                            shuffle = False
                        )
                
                # Eval: Rollout evaluation and visualization
                with TimeProfiler(self.time_report, 'eval'):
                    if self.eval_interval > 0 and (epoch + 1) % self.eval_interval == 0:
                        self.eval(epoch)

            # Logging
            if epoch % self.log_interval == 0:
                # Print logs on screen
                time_summary = self.time_report.print(
                    string_mode = True, 
                    in_second = True
                )
                print_info("-"*100)
                print_info(f"Epoch {epoch}")
                if epoch > 0:
                    print_info("[Train] loss = {:.8f}, itemized = {}".format(
                        avg_train_loss, 
                        format_dict(avg_train_loss_itemized, 8)
                    ))
                for valid_dataset_name in self.valid_datasets.keys():
                    print_info("[Valid] dataset [{}]: loss = {:.8f}, itemized = {}".format(
                        valid_dataset_name, 
                        avg_valid_losses[valid_dataset_name],
                        format_dict(avg_valid_losses_itemized[valid_dataset_name], 8)
                    ))
                print_info("[Time Report] {}".format(time_summary))
                if epoch > 0:
                    print_info("[Grad Info] {}".format(format_dict(grad_info, 3)))
                
                # Logging to tensorboard
                self.logger.add_scalar(
                    'params/lr/epoch', self.lr, epoch)
                    
                if epoch > 0:
                    self.logger.add_scalar(
                        'training/train_loss/epoch', 
                        avg_train_loss, 
                        epoch
                    )
                    self.logger.add_scalar(
                        'training/gradients_before_clip/epoch', 
                        grad_info['grad_norm_before_clip'], 
                        epoch
                    )

                    if self.truncate_grad:
                        self.logger.add_scalar(
                            'training/gradients_after_clip/epoch', 
                            grad_info['grad_norm_after_clip'], 
                            epoch
                        )

                for valid_dataset_name in self.valid_datasets.keys():
                    self.logger.add_scalar(
                        f'training/valid_{valid_dataset_name}_loss/epoch', 
                        avg_valid_losses[valid_dataset_name], 
                        epoch
                    )

                if epoch > 0:
                    for key in avg_train_loss_itemized:
                        self.logger.add_scalar(
                            f'training_info/{key}/epoch', 
                            avg_train_loss_itemized[key], 
                            epoch
                        )
                            
                for valid_dataset_name in self.valid_datasets.keys():
                    for key in avg_valid_losses_itemized[valid_dataset_name]:
                        self.logger.add_scalar(
                            f'validating_info/{key}_{valid_dataset_name}/epoch', 
                            avg_valid_losses_itemized[valid_dataset_name][key], 
                            epoch
                        )

                self.logger.flush()
            
            # Saving model
            if self.save_interval > 0 and (epoch + 1) % self.save_interval == 0:
                self.save_model("model_epoch{}"
                                .format(epoch))
            
            for valid_dataset_name in self.valid_datasets.keys():
                if avg_valid_losses[valid_dataset_name] < best_valid_losses[valid_dataset_name]:
                    best_valid_losses[valid_dataset_name] = avg_valid_losses[valid_dataset_name]
                    self.save_model('best_valid_{}_model'.format(valid_dataset_name))
                    with open(os.path.join(
                            self.model_log_dir, 
                            f'saved_best_valid_{valid_dataset_name}_model_epochs.txt'
                        ), 'a') as fp:
                        fp.write(f"{epoch}\n")
                    print_ok('Save Best Valid {} Model at Epoch {} with loss {:.8f}.'.format(
                                valid_dataset_name, 
                                epoch, 
                                avg_valid_losses[valid_dataset_name]
                            ))

        self.save_model("final_model")

        self.logger.finish()
            
    @torch.no_grad()
    def eval(self, epoch):
        self.neural_model.eval()
        print_info("-"*100)
        print('Evaluating')
        # eval_error in shape (T, N, state_dim)
        eval_error, eval_trajectories, error_stats = \
            self.evaluator.evaluate_action_mode(
                num_traj = self.num_eval_rollouts,
                eval_mode = 'rollout',
                env_mode = 'neural',
                trajectory_source = self.eval_mode,
                render = self.eval_render,
                passive = self.eval_passive,
            )

        # logging
        for error_metric_name in error_stats['overall'].keys():
            self.logger.add_scalar(
                f'eval_{self.eval_horizon}-steps/{error_metric_name}/epoch',
                error_stats['overall'][error_metric_name],
                epoch
            )
        for error_metric_name in error_stats['step-wise'].keys():
            for i in range(error_stats['step-wise'][error_metric_name].shape[0]):
                self.logger.add_scalar(
                    f'eval_details/{error_metric_name}_step_{i}/epoch',
                    error_stats['step-wise'][error_metric_name][i],
                    epoch
                )
        
        print_white(
            "[Evaluate], Num Rollouts = {}, "
            "Rollout Length = {}, "
            "Rollout MSE Error = {:.8f}, "
            "Rollout MSE Error (joint_q) = {:.8f}"
            .format(
                self.num_eval_rollouts, 
                self.eval_horizon, 
                error_stats['overall']['error(MSE)'],
                error_stats['overall']['q_error(MSE)'],
            )
        )

        if error_stats['overall']['error(MSE)'] < self.best_eval_error:
            self.best_eval_error = error_stats['overall']['error(MSE)']
            self.save_model('best_eval_model')
            print_ok(
                'Save Best Eval Model at Epoch {} with MSE error {}.'
                .format(epoch, error_stats['overall']['error(MSE)'])
            )
            with open(os.path.join(self.model_log_dir, 'saved_best_eval_model_epochs.txt'), 'a') as fp:
                fp.write(f"{epoch}\n")

    def test(self):
        self.time_report = TimeReport(cuda_synchronize = False)
        self.time_report.add_timers([
            'epoch', 'other', 'dataloader', 
            'compute_loss', 'backward', 'eval'
        ])

        self.neural_model.eval()
        
        valid_loaders = {}
        valid_loader_iters = {}
        best_valid_losses = {}
        for valid_dataset_name in self.valid_datasets.keys():
            valid_loaders[valid_dataset_name] = DataLoader(
                dataset = self.valid_datasets[valid_dataset_name],
                batch_size = self.batch_size,
                collate_fn = self.collate_fn,
                shuffle = True,
                num_workers = self.num_data_workers,
                drop_last = True
            )
            valid_loader_iters[valid_dataset_name] = iter(valid_loaders[valid_dataset_name])
            best_valid_losses[valid_dataset_name] = np.inf

        # Valid
        avg_valid_losses, avg_valid_losses_itemized = {}, {}
        for valid_dataset_name in self.valid_datasets.keys():
            num_valid_batches = len(valid_loaders[valid_dataset_name])
            avg_valid_losses[valid_dataset_name], avg_valid_losses_itemized[valid_dataset_name], _ = \
                self.one_epoch(train = False, 
                            dataloader = valid_loaders[valid_dataset_name],
                            dataloader_iter = valid_loader_iters[valid_dataset_name], 
                            num_batches = num_valid_batches,
                            shuffle = False,
                            info = valid_dataset_name)
            print_info("Valid dataset [{}]: loss = {:.8f}, itemized = {}".format(
                valid_dataset_name, 
                avg_valid_losses[valid_dataset_name],
                format_dict(avg_valid_losses_itemized[valid_dataset_name], 8)
            ))

        # Rollout Eval
        print('Evaluating')
        num_eval_rollouts = self.num_eval_rollouts
        eval_error, _ = self.evaluator.evaluate_action_mode(
            num_traj = num_eval_rollouts,
            eval_mode = 'rollout',
            env_mode = 'neural',
            trajectory_source = self.eval_mode,
            render = self.eval_render,
            passive = self.eval_passive,
            silent = True
        )
        
        # Logging
        print_info("--------------------------------------------------")
        print_info(f"Test Summary:")
        for valid_dataset_name in self.valid_datasets.keys():
            print_info("Valid dataset [{}]: loss = {:.8f}, itemized = {}".format(
                valid_dataset_name, 
                avg_valid_losses[valid_dataset_name],
                format_dict(avg_valid_losses_itemized[valid_dataset_name], 8)
            ))
        print_info("--------------------------------------------------")
        print_info("Eval ({} rollouts) Error: {}, Error per step: {}".format(
            num_eval_rollouts, 
            (eval_error ** 2).mean(), 
            (eval_error ** 2).mean((-1, -2))
        ))
        print_info("--------------------------------------------------")

    def save_model(self, filename = None):
        if filename is None:
            filename = 'best_model'
        
        torch.save(
            [self.neural_model, self.neural_env.robot_name], 
            os.path.join(self.model_log_dir, '{}.pt'.format(filename))
        )
    




