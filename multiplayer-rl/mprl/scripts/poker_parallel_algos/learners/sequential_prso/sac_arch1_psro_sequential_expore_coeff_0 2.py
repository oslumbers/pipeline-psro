from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import logging
import os
import shutil
import time
from socket import gethostname

import dill
import numpy as np
import ray
from mprl.utility_services.payoff_table import PolicySpec
from mprl.utility_services.worker import LearnerManagerInterface, ConsoleManagerInterface
from ray import tune
from ray.rllib.utils import try_import_tf
from ray.tune.experiment import DEFAULT_RESULTS_DIR
from mprl.utility_services.cloud_storage import connect_storage_client, maybe_download_object, upload_file
from mprl.rl.envs.opnspl.poker_multiagent_env import PARTIALLY_OBSERVABLE, \
    PokerMultiAgentEnv
from mprl.rl.sac.sac import SACTrainer
from mprl.rl.sac.sac_policy import SACDiscreteTFPolicy
from mprl.scripts.poker_parallel_algos.utils.configs import POKER_TRAINER_BASE_CONFIG, POKER_ENV_CONFIG, \
    POKER_SUBMISSION_IMPROVEMENT_THRESHOLD_PER_TIMESTEPS, POKER_SUBMISSION_IMPROVEMENT_THRESHOLD_REWARD, \
    POKER_SUBMISSION_MAX_TIMESTEPS, POKER_SUBMISSION_MIN_TIMESTEPS, POKER_SUBMISSION_THRESHOLD_STEPS_START, \
    POKER_METANASH_FICTITIOUS_PLAY_ITERS, POKER_GAME_VERSION
from mprl.scripts.poker_parallel_algos.utils.policy_config_keys import POKER_ARCH1_MODEL_CONFIG_KEY
from mprl.scripts.poker_parallel_algos.utils.metanash import get_fp_metanash_for_latest_payoff_table
from mprl.utils import datetime_str, ensure_dir, with_updates
from mprl import TUNE_SAVE_DIR

# log level for our code, not for Ray/rllib
logger = logging.getLogger(__name__)

tf = try_import_tf()

TRAIN_POLICY = "train_policy"
STATIC_POLICY = "static_policy"

OBSERVATION_MODE = PARTIALLY_OBSERVABLE

TRAINER_CLASS = SACTrainer
TRAIN_POLICY_CLASS = SACDiscreteTFPolicy
TRAIN_POLICY_MODEL_CONFIG_KEY = POKER_ARCH1_MODEL_CONFIG_KEY
SUBMISSION_POLICY_TAGS = [f"hostname {gethostname()}"]

STATIC_POLICY_CLASS = SACDiscreteTFPolicy
STATIC_POLICY_MODEL_CONFIG_KEY = POKER_ARCH1_MODEL_CONFIG_KEY

ACCEPTED_OPPONENT_POLICY_CLASSES = [SACDiscreteTFPolicy]
ACCEPTED_OPPONENT_POLICY_CLASS_NAMES = [cls.__name__ for cls in ACCEPTED_OPPONENT_POLICY_CLASSES]
ACCEPTED_OPPONENT_MODEL_CONFIG_KEYS = [POKER_ARCH1_MODEL_CONFIG_KEY]

MANAGER_SEVER_HOST = os.getenv("MANAGER_SERVER_HOST", "localhost")

MANAGER_PORT = int(os.getenv("MANAGER_PORT"))
if not MANAGER_PORT:
    raise ValueError("Environment variable MANAGER_PORT needs to be set.")

PSRO_EXPLORATION_COEFF = 0.0

METANASH_FICTITIOUS_PLAY_ITERS = POKER_METANASH_FICTITIOUS_PLAY_ITERS
SUBMISSION_IMPROVEMENT_THRESHOLD_REWARD = POKER_SUBMISSION_IMPROVEMENT_THRESHOLD_REWARD
SUBMISSION_IMPROVEMENT_THRESHOLD_PER_TIMESTEPS = POKER_SUBMISSION_IMPROVEMENT_THRESHOLD_PER_TIMESTEPS
SUBMISSION_THRESHOLD_STEPS_START = POKER_SUBMISSION_THRESHOLD_STEPS_START
SUBMISSION_MIN_TIMESTEPS = POKER_SUBMISSION_MIN_TIMESTEPS
SUBMISSION_MAX_TIMESTEPS = POKER_SUBMISSION_MAX_TIMESTEPS
CLOUD_PREFIX = os.getenv("CLOUD_PREFIX", "")

if __name__ == "__main__":
    expected_payoff_matrix_size = 0

    while True:
        logging.basicConfig(level=logging.DEBUG)
        logger.info("\n\n\n\n\n__________________________________________\n"
                    f"LAUNCHED FOR {POKER_GAME_VERSION}\n"
                    f"__________________________________________\n\n\n\n\n")

        storage_client = connect_storage_client()

        size_checker = ConsoleManagerInterface(server_host=MANAGER_SEVER_HOST,
                                               port=MANAGER_PORT,
                                               worker_id=f"size_checker_{gethostname()}_pid_{os.getpid()}",
                                               storage_client=storage_client,
                                               minio_bucket_name=BUCKET_NAME)

        while True:
            current_payoff_matrix_size = size_checker.get_size_of_current_payoff_table()
            if current_payoff_matrix_size < expected_payoff_matrix_size:
                logger.info(
                    f"waiting for payoff matrix to reach size {expected_payoff_matrix_size} (currently {current_payoff_matrix_size})...")
                time.sleep(5)
            # elif current_payoff_matrix_size > expected_payoff_matrix_size:
            #     raise ValueError(f"payoff matrix is now larger than expected (expected {expected_payoff_matrix_size}, currently {current_payoff_matrix_size})")
            else:
                break
        expected_payoff_matrix_size = current_payoff_matrix_size + 1

        ray.init(address=os.getenv('RAY_HEAD_NODE'), ignore_reinit_error=True)
        logger.info("Ray Web UI at {}".format(ray.get_webui_url()))

        base_experiment_name = f"{CLOUD_PREFIX}learner_{POKER_GAME_VERSION}_sac_arch1_psro_sequential_explore_coeff_{PSRO_EXPLORATION_COEFF}"
        full_experiment_name = f"{base_experiment_name}_{gethostname()}_pid_{os.getpid()}_{datetime_str()}"
        experiment_save_dir = os.path.join(DEFAULT_RESULTS_DIR, full_experiment_name)


        def init_static_policy_distribution_after_trainer_init_callback(trainer):
            trainer.storage_client = connect_storage_client()

            logger.info("Initializing trainer manager interface")
            trainer.manager_interface = LearnerManagerInterface(server_host=MANAGER_SEVER_HOST,
                                                                port=MANAGER_PORT,
                                                                worker_id=full_experiment_name,
                                                                storage_client=trainer.storage_client,
                                                                minio_bucket_name=BUCKET_NAME)

            selection_probs, payoff_table, payoff_table_key = get_fp_metanash_for_latest_payoff_table(
                manager_interface=trainer.manager_interface,
                fp_iters=METANASH_FICTITIOUS_PLAY_ITERS,
                accepted_opponent_policy_class_names=ACCEPTED_OPPONENT_POLICY_CLASS_NAMES,
                accepted_opponent_model_config_keys=ACCEPTED_OPPONENT_MODEL_CONFIG_KEYS,
                add_payoff_matrix_noise_std_dev=0.0,
                mix_with_uniform_dist_coeff=PSRO_EXPLORATION_COEFF
            )

            if selection_probs is None:
                assert payoff_table is None
                assert payoff_table_key is None
                print("Payoff table is empty so using random weights for static policy.")
            else:
                print(f"Payoff table loaded from {payoff_table_key}")
                print(f"Policy selection probs: {selection_probs}")

            payoff_table_dill_str = dill.dumps(payoff_table)

            def worker_set_static_policy_distribution(worker):
                worker.policy_map[STATIC_POLICY].static_policy_selection_probs = selection_probs
                worker.policy_map[STATIC_POLICY].payoff_table = dill.loads(payoff_table_dill_str)
                worker.policy_map[STATIC_POLICY].current_policy_key = None

            trainer.workers.foreach_worker(worker_set_static_policy_distribution)


        # def init_train_policy_weights_from_static_policy_distribution_after_trainer_init_callback(trainer):
        #     local_static_policy = trainer.workers.local_worker().policy_map[STATIC_POLICY]
        #     local_train_policy = trainer.workers.local_worker().policy_map[TRAIN_POLICY]
        #     if local_static_policy.static_policy_selection_probs is None:
        #         print("Payoff table is empty so using random weights for train policy init.")
        #         local_train_policy.init_tag = "init from random"
        #         return
        #
        #     selected_policy_index = np.random.choice(
        #         a=list(range(len(local_static_policy.static_policy_selection_probs))),
        #         p=local_static_policy.static_policy_selection_probs
        #     )
        #     selected_policy_spec: PolicySpec = local_static_policy.payoff_table.get_policy_for_index(selected_policy_index)
        #     local_train_policy.init_tag = f"full init from {selected_policy_spec.key}"
        #
        #     # may not necessarily be true in all scripts
        #     assert selected_policy_spec.class_name == TRAIN_POLICY_CLASS.__name__
        #     assert selected_policy_spec.config_key == TRAIN_POLICY_MODEL_CONFIG_KEY
        #     storage_client = connect_storage_client(endpoint=MINIO_ENDPOINT,
        #                                         access_key=MINIO_ACCESS_KEY,
        #                                         secret_key=MINIO_SECRET_KEY)
        #     weights_local_path, _ = maybe_download_object(storage_client=storage_client,
        #                                                   bucket_name=BUCKET_NAME,
        #                                                   object_name=selected_policy_spec.key,
        #                                                   force_download=False)
        #
        #     def worker_set_train_policy_weights(worker):
        #         train_policy = worker.policy_map[TRAIN_POLICY]
        #         train_policy.load_model_weights(load_file_path=weights_local_path,
        #                                         add_scope_prefix=TRAIN_POLICY)
        #
        #     trainer.workers.foreach_worker(worker_set_train_policy_weights)

        # def sample_new_static_policy_weights_for_each_worker_on_train_result_callback(params):
        #     trainer = params['trainer']
        #
        #     def worker_sample_new_static_policy(worker):
        #         static_policy = worker.policy_map[STATIC_POLICY]
        #
        #         if static_policy.static_policy_selection_probs is None:
        #             return
        #
        #         selected_policy_index = np.random.choice(
        #             a=list(range(len(static_policy.static_policy_selection_probs))),
        #             p=static_policy.static_policy_selection_probs
        #         )
        #         selected_policy_spec: PolicySpec = static_policy.payoff_table.get_policy_for_index(selected_policy_index)
        #         assert selected_policy_spec.class_name in ACCEPTED_OPPONENT_POLICY_CLASS_NAMES
        #         assert selected_policy_spec.config_key in ACCEPTED_OPPONENT_MODEL_CONFIG_KEYS
        #
        #         if static_policy.current_policy_key != selected_policy_spec.key:
        #             # print(f"sampled policy {selected_policy_spec.key} (loading weights)")
        #             storage_client = connect_storage_client(endpoint=MINIO_ENDPOINT,
        #                                                 access_key=MINIO_ACCESS_KEY,
        #                                                 secret_key=MINIO_SECRET_KEY)
        #             weights_local_path, _ = maybe_download_object(storage_client=storage_client,
        #                                                           bucket_name=BUCKET_NAME,
        #                                                           object_name=selected_policy_spec.key,
        #                                                           force_download=False)
        #             static_policy.load_model_weights(load_file_path=weights_local_path,
        #                                              add_scope_prefix=STATIC_POLICY)
        #             static_policy.current_policy_key = selected_policy_spec.key
        #         # else:
        #         #     print(f"sampled policy {selected_policy_spec.key} (weights already loaded)")
        #
        #     trainer.workers.foreach_worker(worker_sample_new_static_policy)

        def sample_new_static_policy_weights_for_each_worker_on_episode_start(params):
            policies = params['policy']

            static_policy = policies[STATIC_POLICY]

            if static_policy.static_policy_selection_probs is None:
                return

            selected_policy_index = np.random.choice(
                a=list(range(len(static_policy.static_policy_selection_probs))),
                p=static_policy.static_policy_selection_probs
            )
            selected_policy_spec: PolicySpec = static_policy.payoff_table.get_policy_for_index(selected_policy_index)
            assert selected_policy_spec.class_name in ACCEPTED_OPPONENT_POLICY_CLASS_NAMES
            assert selected_policy_spec.config_key in ACCEPTED_OPPONENT_MODEL_CONFIG_KEYS

            if static_policy.current_policy_key != selected_policy_spec.key:
                # print(f"sampled policy {selected_policy_spec.key} (loading weights)")
                storage_client = connect_storage_client()
                weights_local_path, _ = maybe_download_object(storage_client=storage_client,
                                                              bucket_name=BUCKET_NAME,
                                                              object_name=selected_policy_spec.key,
                                                              force_download=False)
                static_policy.load_model_weights(load_file_path=weights_local_path,
                                                 add_scope_prefix=STATIC_POLICY)
                static_policy.current_policy_key = selected_policy_spec.key


        def stop_and_submit_if_not_improving_on_train_result_callback(params):
            trainer = params['trainer']
            result = params['result']
            result['stop_signal'] = False

            should_submit = False
            submit_reason = None

            if not hasattr(trainer, 'previous_threshold_check_reward'):
                trainer.previous_threshold_check_reward = -100.0
                trainer.next_threshold_check_timesteps = SUBMISSION_IMPROVEMENT_THRESHOLD_PER_TIMESTEPS + SUBMISSION_THRESHOLD_STEPS_START
                print(f"fist threshold check at {trainer.next_threshold_check_timesteps} timesteps")

            if result['timesteps_total'] >= SUBMISSION_THRESHOLD_STEPS_START and \
                    SUBMISSION_IMPROVEMENT_THRESHOLD_PER_TIMESTEPS is not None and \
                    SUBMISSION_IMPROVEMENT_THRESHOLD_REWARD is not None:

                if result['timesteps_total'] >= trainer.next_threshold_check_timesteps:
                    trainer.next_threshold_check_timesteps = max(
                        trainer.next_threshold_check_timesteps + SUBMISSION_IMPROVEMENT_THRESHOLD_PER_TIMESTEPS,
                        result['timesteps_total'] + 1)

                    target_reward = trainer.previous_threshold_check_reward + SUBMISSION_IMPROVEMENT_THRESHOLD_REWARD
                    result['target_reward'] = target_reward
                    measured_reward = result['policy_reward_mean'][TRAIN_POLICY]
                    print(
                        f"{result['timesteps_total']} timesteps: {TRAIN_POLICY} reward: {measured_reward}, target reward: {target_reward}")

                    if measured_reward < target_reward and \
                            (SUBMISSION_MIN_TIMESTEPS is None or result['timesteps_total'] >= SUBMISSION_MIN_TIMESTEPS):
                        should_submit = True
                        submit_reason = f"plateaued at {measured_reward} reward"
                        print(
                            f"{result['timesteps_total']} timesteps: {TRAIN_POLICY} didn\'t reach target reward. Submitting policy.")
                    else:
                        print(f"next threshold check at {trainer.next_threshold_check_timesteps} timesteps")

                    trainer.previous_threshold_check_reward = measured_reward

            if SUBMISSION_MAX_TIMESTEPS is not None and result['timesteps_total'] >= SUBMISSION_MAX_TIMESTEPS:
                should_submit = True
                submit_reason = f"hit max timesteps of {SUBMISSION_MAX_TIMESTEPS}"
                print(f"Trainer hit max timesteps. Submitting policy.")

            if should_submit:
                assert submit_reason is not None
                result['stop_signal'] = True
                local_train_policy = trainer.workers.local_worker().policy_map[TRAIN_POLICY]

                tags = [*SUBMISSION_POLICY_TAGS,
                        submit_reason,
                        f"timesteps: {result['timesteps_total']}",
                        f"episodes: {result['episodes_total']}"]
                if hasattr(local_train_policy, "init_tag"):
                    tags += local_train_policy.init_tag

                checkpoints_dir = os.path.join(experiment_save_dir, "policy_submissions")
                checkpoint_name = f"{datetime_str()}_iter_{result['training_iteration']}.dill"
                checkpoint_save_path = os.path.join(checkpoints_dir, checkpoint_name)
                local_train_policy.save_model_weights(save_file_path=checkpoint_save_path,
                                                      remove_scope_prefix=TRAIN_POLICY)
                policy_key = os.path.join(base_experiment_name, full_experiment_name,
                                          "policy_submissions", checkpoint_name)
                storage_client = connect_storage_client()
                upload_file(storage_client=storage_client,
                            bucket_name=BUCKET_NAME,
                            object_key=policy_key,
                            local_source_path=checkpoint_save_path)
                trainer.manager_interface.submit_new_policy_for_population(
                    policy_weights_key=policy_key,
                    policy_config_key=TRAIN_POLICY_MODEL_CONFIG_KEY,
                    policy_class_name=TRAIN_POLICY_CLASS.__name__,
                    policy_tags=tags
                )


        train_model_config_local_file_path, _ = maybe_download_object(storage_client=storage_client,
                                                                      bucket_name=BUCKET_NAME,
                                                                      object_name=TRAIN_POLICY_MODEL_CONFIG_KEY)
        with open(train_model_config_local_file_path, 'r') as config_file:
            train_model_config = json.load(fp=config_file)

        static_model_config_local_file_path, _ = maybe_download_object(storage_client=storage_client,
                                                                       bucket_name=BUCKET_NAME,
                                                                       object_name=STATIC_POLICY_MODEL_CONFIG_KEY)
        with open(static_model_config_local_file_path, 'r') as config_file:
            static_model_config = json.load(fp=config_file)


        def train_policy_mapping_fn(agent_id):
            if agent_id == 1:
                return TRAIN_POLICY
            elif agent_id == 0 or agent_id == -1:
                return STATIC_POLICY
            else:
                raise ValueError("train_policy_mapping_fn: wasn't expecting an agent_id other than 1 or -1")


        temp_env = ENV_CLASS(POKER_ENV_CONFIG)
        obs_space = temp_env.observation_space
        act_space = temp_env.action_space

        trainer_config = with_updates(base_dict=POKER_TRAINER_BASE_CONFIG, updates_dict={
            "multiagent": {
                "policies": {
                    TRAIN_POLICY: (TRAIN_POLICY_CLASS, obs_space, act_space, {
                        'model': train_model_config,
                    }),
                    STATIC_POLICY: (STATIC_POLICY_CLASS, obs_space, act_space, {
                        'model': static_model_config,
                    }),
                },
                "policy_mapping_fn": train_policy_mapping_fn,
                "policies_to_train": [TRAIN_POLICY],
            },

            "callbacks_after_trainer_init": [
                init_static_policy_distribution_after_trainer_init_callback,
            ],
            "callbacks": {
                "on_train_result": stop_and_submit_if_not_improving_on_train_result_callback,
                'on_episode_start': sample_new_static_policy_weights_for_each_worker_on_episode_start,
            },
        })

        # save running script to file
        current_code_file_path = os.path.abspath(__file__)
        copy_code_to_path = os.path.join(experiment_save_dir, "launch_script.py")
        ensure_dir(copy_code_to_path)
        shutil.copy2(src=current_code_file_path, dst=copy_code_to_path, follow_symlinks=True)


        def trial_name_creator(trial):
            config = trial.config
            return "sac_learner"


        analysis = tune.run(
            local_dir=TUNE_SAVE_DIR,
            name=full_experiment_name,
            # upload_dir=base_experiment_name,
            # sync_to_cloud=get_tune_sync_to_cloud_fn(storage_client=storage_client, bucket_name=BUCKET_NAME),
            checkpoint_at_end=False,
            keep_checkpoints_num=0,
            checkpoint_freq=0,
            num_samples=1,
            max_failures=0,
            reuse_actors=False,
            trial_name_creator=trial_name_creator,
            export_formats=[],
            stop={"stop_signal": True},
            run_or_experiment=TRAINER_CLASS,
            config=trainer_config)

        print("Experiment Done!")
