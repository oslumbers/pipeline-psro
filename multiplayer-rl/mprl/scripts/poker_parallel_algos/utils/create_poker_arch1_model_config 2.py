from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os

import ray
from ray.rllib.utils import try_import_tf
from ray.tune.experiment import DEFAULT_RESULTS_DIR

from mprl.utility_services.cloud_storage import connect_storage_client, upload_file, key_exists
from mprl.rl.envs.opnspl.poker_multiagent_env import POKER_ENV, KUHN_POKER, LEDUC_POKER, PARTIALLY_OBSERVABLE, PokerMultiAgentEnv

from mprl.rl.common.sac_stratego_model import SAC_STRATEGO_MODEL
from mprl.rl.common.stratego_preprocessor import STRATEGO_PREPROCESSOR
from mprl.rl.sac.sac import SACTrainer
from mprl.rl.sac.sac_policy import SACDiscreteTFPolicy
from mprl.scripts.poker_parallel_algos.utils.policy_config_keys import POKER_ARCH1_MODEL_CONFIG_KEY
from mprl.utils import datetime_str, with_updates

# log level for our code, not for Ray/rllib
logger = logging.getLogger(__name__)

tf = try_import_tf()

TRAIN_POLICY = "train_policy"
STATIC_POLICY = "static_policy"

POKER_GAME_VERSION = KUHN_POKER
OBSERVATION_MODE = PARTIALLY_OBSERVABLE

TRAINER_CLASS = SACTrainer
TRAIN_POLICY_CLASS = SACDiscreteTFPolicy

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ray.init(address=os.getenv('RAY_HEAD_NODE'))
    logger.info("Ray Web UI at {}".format(ray.get_webui_url()))

    base_experiment_name = f"create_{POKER_GAME_VERSION}_sac_arch1_policy_config"
    full_experiment_name = f"{base_experiment_name}_{datetime_str()}"
    experiment_save_dir = os.path.join(DEFAULT_RESULTS_DIR, full_experiment_name)

    storage_client = connect_storage_client()

    poker_env_config = {
        'version': POKER_GAME_VERSION,
    }

    temp_env = PokerMultiAgentEnv(poker_env_config)
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space

    model_config = {
        "custom_preprocessor": STRATEGO_PREPROCESSOR,
        "custom_model": SAC_STRATEGO_MODEL,
        "vf_share_layers": False,
        "conv_filters": [],
        "fcnet_hiddens": [40, 40, 40],
        "custom_options": {
            # "final_pi_filter_amt": 100,
            # "lstm_filters": [],
            "mask_invalid_actions": True,
            "observation_mode": OBSERVATION_MODE,
            "q_fn": True,
            "fake_lstm": False,
            "use_lstm": False,
        },
    }


    def train_policy_mapping_fn(agent_id):
        if agent_id == 1:
            return TRAIN_POLICY
        elif agent_id == -1:
            return STATIC_POLICY
        else:
            raise ValueError("train_policy_mapping_fn: wasn't expecting an agent_id other than 1 or -1")


    total_gpus_for_trial = 1
    num_gpus_for_opt = 1
    num_workers = 0

    trainer_config = {
        "log_level": "INFO",
        "metrics_smoothing_episodes": 10000,
        "memory_per_worker": 7019430400,
        "num_envs_per_worker": 1,
        "num_workers": num_workers,
        "num_gpus_per_worker": 0,
        "env": POKER_ENV,
        "env_config": with_updates(base_dict=poker_env_config, updates_dict={
            'num_envs': 64,
        }),

        "multiagent": {
            "policies": {
                TRAIN_POLICY: (SACDiscreteTFPolicy, obs_space, act_space, {
                    'model': model_config,
                }),
                STATIC_POLICY: (SACDiscreteTFPolicy, obs_space, act_space, {
                    'model': model_config,
                }),
            },
            "policy_mapping_fn": train_policy_mapping_fn,
            "policies_to_train": [TRAIN_POLICY],
        },

        "callbacks_after_trainer_init": [
            lambda trainer: trainer.save_policy_model_configs_to_json(),
        ],
        "gamma": 0.99,
        "train_batch_size": 2048,
        "optimization": {
            "actor_learning_rate": 1e-4,
            "critic_learning_rate": 1e-4,
            "entropy_learning_rate": 1e-3,
        },
        "max_entropy_target_proportion": 0.3,
        "batch_mode": 'truncate_episodes',
        "num_gpus": num_gpus_for_opt,
        "sample_batch_size": 100,
    }

    trainer = TRAINER_CLASS(config=trainer_config)

    if key_exists(storage_client=storage_client, bucket_name="stratego", object_name=POKER_ARCH1_MODEL_CONFIG_KEY):
        response = input(f"\n\nThe key {POKER_ARCH1_MODEL_CONFIG_KEY} already exists. "
                         f"\nDo you really want to override it?"
                         f"\nOther programs may already by using this key."
                         f"\nEnter 'y' to overwrite it.\n")
        if response != 'y':
            print("exiting...")
            exit(0)
        response = input(f"Are you REALLY sure you want to overwrite {POKER_ARCH1_MODEL_CONFIG_KEY}?????"
                         f"\nEnter 'y' to overwrite it (last warning).\n")
        if response != 'y':
            print("exiting...")
            exit(0)

    upload_file(storage_client=storage_client,
                bucket_name="stratego",
                object_key=POKER_ARCH1_MODEL_CONFIG_KEY,
                local_source_path=os.path.join(trainer._logdir, f"{TRAIN_POLICY}_config.json"))

    logger.info("DONE!")
