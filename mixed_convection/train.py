import time
import numpy as np
import jax
from tqdm.notebook import tqdm
import os
from Tam_project_2.jaxpi2.models_1 import create_lr_schedule, create_optimizer, create_arch, create_train_state
from jaxpi2.samplers import UniformSampler, MeshSampler
from jaxpi2.checkpointing import create_checkpoint_manager
from jaxpi2.logging import Logger
from jaxpi2.checkpointing import save_checkpoint
from jaxpi2.utils import create_update_scheduler
from mixed_convection import models
from mixed_convection.config import base_line
import pickle

def train_model(config, lb, ub, samplers):
    ckpt_path = os.path.join(os.getcwd(), config.wandb.name, "ckpt")
    history_path = os.path.join(os.getcwd(), config.wandb.name,"history.pkl")
    ckpt_mngr = create_checkpoint_manager(config.saving, ckpt_path)
    
    lr = create_lr_schedule(config.optim)
    tx = create_optimizer(config.optim, lr)
    arch = create_arch(config.arch)
    state = create_train_state(config, tx, arch)

    model = models.MixedConvectionPorous(config, lr, tx, arch, state, lb, ub)
    init_state = model.state  
    history = {'loss': []}    

    freq_pts = config.pseudo_time.update_schedule.get('every', 100) if hasattr(config.pseudo_time, 'update_schedule') else 100
    freq_loss = config.loss_weighting.update_schedule.get('every', 100) if hasattr(config.loss_weighting, 'update_schedule') else 100
    freq_log = config.logging.get('log_every_steps', 1000)

    pbar = tqdm(range(config.training.max_steps), desc="Training PINN")

    print("Bắt đầu huấn luyện...")
    start_time = time.time()


    for step in pbar:

        batch = {}
        for key, sampler in samplers.items():
            batch[key] = next(sampler)
        
        # BƯỚC 1: Forward & Update
        model.state, loss, loss_dict = model.step(model.state, batch)
        if config.pseudo_time.enabled and config.pseudo_time.strategy == "dynamic":
            if step % freq_pts == 0 and config.pseudo_time.update_schedule.start <= step:
                model.state = model.update_pts_weights(model.state, init_state, batch['eqn'])


        if config.loss_weighting.strategy == "dynamic":
            if step % freq_loss == 0:
                model.state = model.update_loss_weights(model.state, batch)

        # BƯỚC 4: Logging
        if jax.process_index() == 0:
            if step % freq_log == 0:
                val_loss = np.asarray(loss).item()
                history['loss'].append(val_loss)
                
                for k, v in loss_dict.items():
                    if k not in history:
                        history[k] = []
                    history[k].append(np.asarray(v).item())
                
                if hasattr(model.state, 'pts_weights') and model.state.pts_weights is not None:
                    for k, v in model.state.pts_weights.items():
                        hist_key = f"pts_weight_{k}"
                        
                        # 1. Kiểm tra và khởi tạo list nếu chưa có
                        if hist_key not in history:
                            history[hist_key] = []

                        history[hist_key].append(np.asarray(v).item())

                pbar.set_postfix({'Loss': f"{val_loss:.4e}"})
                if step > 0 and step % (freq_log * 5) == 0:
                    elapsed = time.time() - start_time
                    details_str = " | ".join([f"{k}: {np.asarray(v).item():.4e}" for k, v in loss_dict.items()])
                    pts_str = ""
                    if hasattr(model.state, 'pts_weights') and model.state.pts_weights is not None:
                        pts_str = " | " + " | ".join([f"tau_{k}: {np.asarray(v).item():.4e}" for k, v in model.state.pts_weights.items()])

                    print(f"Step {step:05d} | Loss: {val_loss:.4e} | {details_str}| {details_str}{pts_str} | Time/Log: {elapsed:.2f}s")
                    
                    start_time = time.time() 

            if config.saving.save_every_steps > 0:
                if (step + 1) % config.saving.save_every_steps == 0 or (step + 1) == config.training.max_steps:
                    save_checkpoint(ckpt_mngr, model.state)

    with open(history_path, 'wb') as f:
        pickle.dump(history, f)

    print("Done training")
    if config.saving.save_every_steps > 0 and jax.process_index() == 0:
        ckpt_mngr.wait_until_finished()
    return model, history

