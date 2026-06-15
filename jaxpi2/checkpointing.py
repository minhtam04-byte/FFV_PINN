import os
import json
from flax import serialization
import jax.numpy as jnp
import orbax.checkpoint as ocp
import jax
import numpy as np 
import ml_collections

def create_checkpoint_manager(config, ckpt_path, suffix=None):
    if suffix is not None:
        ckpt_path = os.path.join(ckpt_path, str(suffix))

    ckpt_options = ocp.CheckpointManagerOptions(
        max_to_keep=config.num_keep_ckpts,
        create=True,
    )
    checkpointer = ocp.PyTreeCheckpointer()
    ckpt_mngr = ocp.CheckpointManager(ckpt_path, checkpointers=checkpointer, options=ckpt_options)
    return ckpt_mngr


# def save_checkpoint(ckpt_mngr, state):
#     ckpt_mngr.save(state.step, args=ocp.args.StandardSave(state))


# def restore_checkpoint(ckpt_mngr, state, step=None):
#     step = step if step is not None else ckpt_mngr.latest_step()
#     restored = ckpt_mngr.restore(
#         step,
#         args=ocp.args.StandardRestore(state),
#     )
#     return restored
def sanitize_dict_for_orbax(obj):
    # Đệ quy đi sâu vào các nhánh của từ điển/danh sách
    if isinstance(obj, dict):
        return {k: sanitize_dict_for_orbax(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_dict_for_orbax(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(sanitize_dict_for_orbax(v) for v in obj)
    
    # KẺ HỦY DIỆT WEAK_TYPE:
    # Bất kể là JAX Array (weak/strong) hay số float/int, 
    # Cứ gộp tất cả thành Numpy Array tiêu chuẩn!
    elif hasattr(obj, 'dtype') and hasattr(obj, 'shape'):
        return np.array(obj)
    elif isinstance(obj, (int, float, bool)):
        return np.array(obj)
    
    return obj

def save_checkpoint(ckpt_mngr, state):
    current_step = int(np.asarray(state.step))
    
    # 1. Trải phẳng TrainState thành dạng Dict Python
    state_dict = serialization.to_state_dict(state)
    
    # 2. Ép toàn bộ về Numpy Array để Orbax không bị lỗi dtype
    safe_state_dict = sanitize_dict_for_orbax(state_dict)
    
    # 3. Lưu xuống ổ cứng
    ckpt_mngr.save(current_step, items=safe_state_dict)

def restore_checkpoint(ckpt_mngr, state, step=None):
    step = step if step is not None else ckpt_mngr.latest_step()
    
    # 1. Tạo "bộ khung" Dict an toàn giống như lúc lưu
    state_dict = serialization.to_state_dict(state)
    safe_state_dict = sanitize_dict_for_orbax(state_dict)
    
    # 2. Khôi phục từ ổ cứng lên
    restored_dict = ckpt_mngr.restore(step, items=safe_state_dict)
    
    # 3. Ép ngược lại vào cấu trúc TrainState của JAX để dùng tiếp
    return serialization.from_state_dict(state, restored_dict)

""" 
from jaxpi2.checkpointing import (
    create_checkpoint_manager,
    save_checkpoint,
    restore_checkpoint,
)
lr_2 = create_lr_schedule(config.optim)
tx_2 = create_optimizer(config.optim, lr_2)
arch_2 = create_arch(config.arch)
state_2 = create_train_state(config, tx_2, arch_2)
model2 = models.MixedConvectionPorous(config, lr, tx_2, arch_2, state_2, lb, ub)
ckpt_path_2 = os.path.join(os.getcwd(), config.wandb.name, "ckpt")
ckpt_mngr_2 = create_checkpoint_manager(config.saving, ckpt_path)
state_2 = restore_checkpoint(ckpt_mngr_2, state_2)
"""


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        # Custom serialization for JAX numpy arrays
        if isinstance(obj, jnp.ndarray):
            return obj.tolist()  # Convert JAX numpy array to a list
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)


def save_config(config, workdir, name=None):
    # Create the workdir if it doesn't exist.
    if not os.path.isdir(workdir):
        os.makedirs(workdir)

    # Set default name if not provided
    if name is None:
        name = "config"
    # Correctly append the '.json' extension to the filename
    config_path = os.path.join(workdir, name + '.json')

    # Write the config to a JSON file
    with open(config_path, 'w') as config_file:
        json.dump(config.to_dict(), config_file, cls=CustomJSONEncoder, indent=4)