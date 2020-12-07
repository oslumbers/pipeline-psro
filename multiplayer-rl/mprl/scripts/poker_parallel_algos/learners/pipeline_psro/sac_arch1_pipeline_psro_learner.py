from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import logging
import os
import shutil
import time
from dill import load, dump
import copy

import dill
import numpy as np
import ray
from ray.rllib.utils.memory import ray_get_and_free

from termcolor import colored
from mprl.utility_services.payoff_table import PolicySpec
from mprl.utility_services.worker import LearnerManagerInterface, ConsoleManagerInterface
from mprl.utility_services.lock_server.lock_client_interface import LockServerInterface
from ray import tune
from ray.rllib.utils import try_import_tf
from ray.tune.experiment import DEFAULT_RESULTS_DIR

from mprl.utility_services.cloud_storage import connect_storage_client, maybe_download_object, get_tune_sync_to_cloud_fn, \
    upload_file, BUCKET_NAME
from mprl.rl.envs.opnspl.poker_multiagent_env import POKER_ENV, KUHN_POKER, LEDUC_POKER, PARTIALLY_OBSERVABLE, PokerMultiAgentEnv
from mprl.rl.common.sac_stratego_model import SAC_STRATEGO_MODEL

from mprl.scripts.poker_parallel_algos.utils.policy_config_keys import POKER_ARCH1_MODEL_CONFIG_KEY
from mprl.scripts.poker_parallel_algos.utils.live_policy_payoff_tracker import LivePolicyPayoffTracker
from mprl.rl.sac.sac import SACTrainer
from mprl.rl.sac.sac_policy import SACDiscreteTFPolicy
from mprl.scripts.poker_parallel_algos.utils.metanash import get_fp_metanash_for_payoff_table
from mprl.utils import datetime_str, ensure_dir, with_updates
from socket import gethostname
from mprl.scripts.poker_parallel_algos.utils.configs import POKER_TRAINER_BASE_CONFIG, POKER_ENV_CONFIG, \
    POKER_SUBMISSION_IMPROVEMENT_THRESHOLD_PER_STEPS, POKER_SUBMISSION_IMPROVEMENT_THRESHOLD_REWARD, \
    POKER_SUBMISSION_MAX_STEPS, POKER_SUBMISSION_MIN_STEPS, POKER_SUBMISSION_THRESHOLD_STEPS_START, \
    POKER_METANASH_FICTITIOUS_PLAY_ITERS, POKER_GAME_VERSION, POKER_PSRO_EXPLORATION_COEFF, \
    POKER_PIPELINE_WARMUP_ENTROPY_TARGET_PROPORTION, POKER_PIPELINE_LIVE_PAYOFF_TABLE_CALC_IS_ASYNCHRONOUS, \
    POKER_PIPELINE_CHECKPOINT_AND_REFRESH_LIVE_TABLE_EVERY_N_STEPS, POKER_PAYOFF_MATRIX_NOISE_STD_DEV, ENV_CLASS, \
    SELECTED_CONFIG_KEY, POKER_PIPELINE_INIT_FROM_POP
from mprl import TUNE_SAVE_DIR

# log level for our code, not for Ray/rllib
logger = logging.getLogger(__name__)

tf = try_import_tf()

TRAIN_POLICY = "train_policy"
STATIC_POLICY = "static_policy"

EVO_UPDATE_FREQ = 100  # [10, 100, 1000]
EVO_WKS = 10  # [10, 100, 1000]
EVO_SIGMA = 0.05  # [0.01, 0.1, 0.5]
EVO_LR = 0.05  # [0.01, 0.5, 0.5]


OBSERVATION_MODE = PARTIALLY_OBSERVABLE

TRAINER_CLASS = SACTrainer
TRAIN_POLICY_CLASS = SACDiscreteTFPolicy
TRAIN_POLICY_MODEL_CONFIG_KEY = SELECTED_CONFIG_KEY
SUBMISSION_POLICY_TAGS = [f"hostname {gethostname()}"]

STATIC_POLICY_CLASS = SACDiscreteTFPolicy
STATIC_POLICY_MODEL_CONFIG_KEY = SELECTED_CONFIG_KEY

ACCEPTED_OPPONENT_POLICY_CLASSES = [SACDiscreteTFPolicy]
ACCEPTED_OPPONENT_POLICY_CLASS_NAMES = [cls.__name__ for cls in ACCEPTED_OPPONENT_POLICY_CLASSES]
ACCEPTED_OPPONENT_MODEL_CONFIG_KEYS = [SELECTED_CONFIG_KEY]

LOCK_SERVER_HOST = os.getenv("LOCK_SERVER_HOST", 'localhost')

MANAGER_SERVER_HOST = os.getenv("MANAGER_SERVER_HOST", "localhost")

MANAGER_PORT = int(os.getenv("MANAGER_PORT", 27003))
if not MANAGER_PORT:
    raise ValueError("Environment variable MANAGER_PORT needs to be set.")

LOCK_SERVER_PORT = os.getenv("LOCK_SERVER_PORT", 27503)
if not LOCK_SERVER_PORT:
    raise ValueError("Environment variable LOCK_SERVER_PORT needs to be set.")

PSRO_EXPLORATION_COEFF = POKER_PSRO_EXPLORATION_COEFF
METANASH_FICTITIOUS_PLAY_ITERS = POKER_METANASH_FICTITIOUS_PLAY_ITERS
SUBMISSION_IMPROVEMENT_THRESHOLD_REWARD = POKER_SUBMISSION_IMPROVEMENT_THRESHOLD_REWARD
SUBMISSION_IMPROVEMENT_THRESHOLD_PER_STEPS = POKER_SUBMISSION_IMPROVEMENT_THRESHOLD_PER_STEPS
SUBMISSION_THRESHOLD_STEPS_START = POKER_SUBMISSION_THRESHOLD_STEPS_START
SUBMISSION_MIN_STEPS = POKER_SUBMISSION_MIN_STEPS
SUBMISSION_MAX_STEPS = POKER_SUBMISSION_MAX_STEPS

PIPELINE_LIVE_PAYOFF_TABLE_CALC_IS_ASYNCHRONOUS = POKER_PIPELINE_LIVE_PAYOFF_TABLE_CALC_IS_ASYNCHRONOUS
PIPELINE_WARMUP_ENTROPY_TARGET_PROPORTION = POKER_PIPELINE_WARMUP_ENTROPY_TARGET_PROPORTION
PIPELINE_COOLDOWN_ENTROPY_TARGET_PROPORTION = POKER_TRAINER_BASE_CONFIG['max_entropy_target_proportion']
PAYOFF_MATRIX_NOISE_STD_DEV = POKER_PAYOFF_MATRIX_NOISE_STD_DEV

CHECKPOINT_AND_REFRESH_LIVE_TABLE_EVERY_N_STEPS = POKER_PIPELINE_CHECKPOINT_AND_REFRESH_LIVE_TABLE_EVERY_N_STEPS
CANT_SUBMIT_UNTIL_LOWER_POLICIES_FINISH = True
INIT_FROM_POPULATION = POKER_PIPELINE_INIT_FROM_POP
CLOUD_PREFIX = os.getenv("CLOUD_PREFIX", "")

if __name__ == "__main__":
    while True:
        logging.basicConfig(level=logging.DEBUG)
        logger.info("\n\n\n\n\n__________________________________________\n"
                    f"LAUNCHED FOR {POKER_GAME_VERSION}\n"
                    f"__________________________________________\n\n\n\n\n")
        print(f"{colored(f'############ HEREEEEEE1 #############', 'magenta')}\n")
        time.sleep(1)


        storage_client = connect_storage_client()

        size_checker = ConsoleManagerInterface(server_host=MANAGER_SERVER_HOST,
                                               port=MANAGER_PORT,
                                               worker_id=f"size_checker_{gethostname()}_pid_{os.getpid()}",
                                               storage_client=storage_client,
                                               minio_bucket_name=BUCKET_NAME)

        ray.init(address=os.getenv('RAY_HEAD_NODE'), ignore_reinit_error=True, log_to_driver=True)
        logger.info("Ray Web UI at {}".format(ray.get_webui_url()))

        base_experiment_name = f"{CLOUD_PREFIX}learner_{POKER_GAME_VERSION}_sac_arch1_pipeline_psro"
        full_experiment_name = f"{base_experiment_name}_{gethostname()}_pid_{os.getpid()}_{datetime_str()}"
        experiment_save_dir = os.path.join(DEFAULT_RESULTS_DIR, full_experiment_name)

        def claim_new_active_policy_after_trainer_init_callback(trainer):
            def set_train_policy_warmup_target_entropy_proportion(worker):
                worker.policy_map[TRAIN_POLICY].set_target_entropy_proportion(PIPELINE_WARMUP_ENTROPY_TARGET_PROPORTION)
            trainer.workers.foreach_worker(set_train_policy_warmup_target_entropy_proportion)

            trainer.storage_client = connect_storage_client()

            logger.info("Initializing trainer manager interface")
            trainer.manager_interface = LearnerManagerInterface(server_host=MANAGER_SERVER_HOST,
                                                                port=MANAGER_PORT,
                                                                worker_id=full_experiment_name,
                                                                storage_client=trainer.storage_client,
                                                                minio_bucket_name=BUCKET_NAME)

            trainer.live_table_tracker = LivePolicyPayoffTracker.remote(
                minio_bucket=BUCKET_NAME,
                manager_host=MANAGER_SERVER_HOST,
                manager_port=MANAGER_PORT,
                lock_server_host=LOCK_SERVER_HOST,
                lock_server_port=LOCK_SERVER_PORT,
                worker_id=full_experiment_name,
                policy_class_name=TRAIN_POLICY_CLASS.__name__,
                policy_config_key=TRAIN_POLICY_MODEL_CONFIG_KEY,
                provide_payoff_barrier_sync=not PIPELINE_LIVE_PAYOFF_TABLE_CALC_IS_ASYNCHRONOUS
            )
            trainer.claimed_policy_num = ray_get_and_free(trainer.live_table_tracker.get_claimed_policy_num.remote())
            trainer.are_all_lower_policies_finished = False
            trainer.payoff_table_needs_update_started = False
            trainer.payoff_table = None
            _do_live_policy_checkpoint(trainer=trainer, training_iteration=0)

            if not PIPELINE_LIVE_PAYOFF_TABLE_CALC_IS_ASYNCHRONOUS:
                # wait for all other learners to also reach this point before continuing
                ray_get_and_free(trainer.live_table_tracker.wait_at_barrier_for_other_learners.remote())

            trainer.new_payoff_table_promise = trainer.live_table_tracker.get_live_payoff_table_dill_pickled.remote(
                first_wait_for_n_seconds=2)
            _process_new_live_payoff_table_result_if_ready(trainer=trainer, block_until_result_is_ready=True)

            if INIT_FROM_POPULATION:
                init_train_policy_weights_from_static_policy_distribution_after_trainer_init_callback(trainer=trainer)
            else:
                print(colored(f"Policy {trainer.claimed_policy_num}: (Initializing train policy to random)", "white"))
            trainer.my_weights_updates = 0.


        def init_train_policy_weights_from_static_policy_distribution_after_trainer_init_callback(trainer):
            local_static_policy = trainer.workers.local_worker().policy_map[STATIC_POLICY]
            local_train_policy = trainer.workers.local_worker().policy_map[TRAIN_POLICY]
            if not hasattr(local_static_policy, 'static_policy_selection_probs') or \
                    local_static_policy.static_policy_selection_probs is None:
                print(colored(f"Policy {trainer.claimed_policy_num}: Payoff table is empty so Initializing train policy to random", "white"))
                local_train_policy.init_tag = "init from random"
                return

            selected_policy_index = np.random.choice(
                a=list(range(len(local_static_policy.static_policy_selection_probs))),
                p=local_static_policy.static_policy_selection_probs
            )
            selected_policy_spec: PolicySpec = local_static_policy.payoff_table.get_policy_for_index(
                selected_policy_index)
            local_train_policy.init_tag = f"full init from {selected_policy_spec.key}"

            # may not necessarily be true in all scripts
            assert selected_policy_spec.class_name == TRAIN_POLICY_CLASS.__name__
            assert selected_policy_spec.config_key == TRAIN_POLICY_MODEL_CONFIG_KEY
            storage_client = connect_storage_client()
            weights_local_path, _ = maybe_download_object(storage_client=storage_client,
                                                          bucket_name=BUCKET_NAME,
                                                          object_name=selected_policy_spec.key,
                                                          force_download=False)

            print(colored(f"Policy {trainer.claimed_policy_num}: Initializing train policy to {selected_policy_spec.key}", "white"))


            # TODO: Here
            def worker_set_train_policy_weights(worker):
                train_policy = worker.policy_map[TRAIN_POLICY]
                train_policy.load_model_weights(load_file_path=weights_local_path,
                                                add_scope_prefix=TRAIN_POLICY)

            trainer.workers.foreach_worker(worker_set_train_policy_weights)


        def _process_new_live_payoff_table_result_if_ready(trainer, block_until_result_is_ready):
            if block_until_result_is_ready:
                timeout = None
            else:
                timeout = 0.0
            payoff_results, _ = ray.wait(trainer.new_payoff_table_promise, timeout=timeout, num_returns=2)
            if block_until_result_is_ready:
                assert len(payoff_results) == 2, f"actual value is {len(payoff_results)}, payoff results: {payoff_results}"

            if len(payoff_results) != 2:
                print(colored(f"Policy {trainer.claimed_policy_num}: waiting for latest live payoff table", "cyan"))
                return

            trainer.new_payoff_table_promise = None

            try:
                new_payoff_dill_in_array, are_all_lower_policies_finished = ray.get(payoff_results)
            except ValueError:
                # todo: remove bandaid against race condition
                return

            assert isinstance(are_all_lower_policies_finished, bool)
            if are_all_lower_policies_finished and not trainer.are_all_lower_policies_finished:
                print(colored(f"Policy {trainer.claimed_policy_num}: All lower policies are finished (this just changed).", "white"))

                def set_train_policy_cooldown_target_entropy_proportion(worker):
                    worker.policy_map[TRAIN_POLICY].set_target_entropy_proportion(PIPELINE_COOLDOWN_ENTROPY_TARGET_PROPORTION)
                trainer.workers.foreach_worker(set_train_policy_cooldown_target_entropy_proportion)

            elif trainer.are_all_lower_policies_finished:
                print(colored(f"Policy {trainer.claimed_policy_num}: All lower policies are finished (and they have been).", "white"))
            elif trainer.are_all_lower_policies_finished and not are_all_lower_policies_finished:
                raise ValueError(f"Policy {trainer.claimed_policy_num}: are_all_lower_policies_finished became False after previously being True")

            trainer.are_all_lower_policies_finished = are_all_lower_policies_finished

            if new_payoff_dill_in_array is not None:
                new_payoff_table = dill.loads(new_payoff_dill_in_array)

                if trainer.payoff_table is None or \
                        set(trainer.payoff_table.get_ordered_keys_in_payoff_matrix()) != set(new_payoff_table.get_ordered_keys_in_payoff_matrix()):
                    trainer.payoff_table = new_payoff_table

                    selection_probs = get_fp_metanash_for_payoff_table(
                        payoff_table=new_payoff_table,
                        fp_iters=METANASH_FICTITIOUS_PLAY_ITERS,
                        accepted_opponent_policy_class_names=ACCEPTED_OPPONENT_POLICY_CLASS_NAMES,
                        accepted_opponent_model_config_keys=ACCEPTED_OPPONENT_MODEL_CONFIG_KEYS,
                        add_payoff_matrix_noise_std_dev=PAYOFF_MATRIX_NOISE_STD_DEV,
                        mix_with_uniform_dist_coeff=PSRO_EXPLORATION_COEFF
                    )

                    if selection_probs is None:
                        assert new_payoff_table is None
                        print(colored(
                            f"Policy {trainer.claimed_policy_num}: Payoff table is empty so using random weights for static policy.",
                            "white"))
                    else:
                        policies_str = ""
                        for policy_key in new_payoff_table.get_ordered_keys_in_payoff_matrix():
                            policies_str += f"{policy_key}"
                        print(colored(
                            f"Policy {trainer.claimed_policy_num}: Payoff Table Policies: {colored(policies_str, 'white')}\n",
                            "white"))
                        print(colored(f"Policy {trainer.claimed_policy_num}: Policy selection probs: {selection_probs}",
                                      "white"))

                    def worker_set_static_policy_distribution(worker):
                        worker.policy_map[STATIC_POLICY].static_policy_selection_probs = selection_probs
                        worker.policy_map[STATIC_POLICY].payoff_table = dill.loads(new_payoff_dill_in_array)
                        worker.policy_map[STATIC_POLICY].current_policy_key = None

                    trainer.workers.foreach_worker(worker_set_static_policy_distribution)
                else:
                    print(colored(f"Policy {trainer.claimed_policy_num}: Not calculating new metanash because keys in the new payoff table are the same as in the current one.", "blue"))

                if PIPELINE_LIVE_PAYOFF_TABLE_CALC_IS_ASYNCHRONOUS and not trainer.are_all_lower_policies_finished:
                    trainer.payoff_table_needs_update_started = True

        def _do_live_policy_checkpoint(trainer, training_iteration):
            local_train_policy = trainer.workers.local_worker().policy_map[TRAIN_POLICY]
            checkpoints_dir = os.path.join(experiment_save_dir, "policy_checkpoints")
            checkpoint_name = f"policy_{trainer.claimed_policy_num}_{datetime_str()}_iter_{training_iteration}.dill"
            checkpoint_save_path = os.path.join(checkpoints_dir, checkpoint_name)
            local_train_policy.save_model_weights(save_file_path=checkpoint_save_path,
                                                  remove_scope_prefix=TRAIN_POLICY)
            policy_key = os.path.join(base_experiment_name, full_experiment_name,
                                      "policy_checkpoints", checkpoint_name)
            storage_client = connect_storage_client()
            upload_file(storage_client=storage_client,
                        bucket_name=BUCKET_NAME,
                        object_key=policy_key,
                        local_source_path=checkpoint_save_path)

            ray_get_and_free(trainer.live_table_tracker.set_latest_key_for_claimed_policy.remote(new_key=policy_key))

        def checkpoint_and_set_static_policy_distribution_on_train_result_callback(params):
            trainer = params['trainer']
            result = params['result']
            result['psro_policy_num'] = trainer.claimed_policy_num
            evo_update(params)

            if not hasattr(trainer, 'next_refresh_steps'):
                trainer.next_refresh_steps = CHECKPOINT_AND_REFRESH_LIVE_TABLE_EVERY_N_STEPS

            if not hasattr(trainer, 'new_payoff_table_promise'):
                trainer.new_payoff_table_promise = None

            if result['timesteps_total'] >= trainer.next_refresh_steps:
                trainer.next_refresh_steps = max(trainer.next_refresh_steps + CHECKPOINT_AND_REFRESH_LIVE_TABLE_EVERY_N_STEPS, result['timesteps_total'] + 1)
                # do checkpoint
                _do_live_policy_checkpoint(trainer=trainer, training_iteration=result['training_iteration'])

                if not PIPELINE_LIVE_PAYOFF_TABLE_CALC_IS_ASYNCHRONOUS:
                    # wait for all other learners to also reach this point before continuing
                    ray_get_and_free(trainer.live_table_tracker.wait_at_barrier_for_other_learners.remote())

                if not trainer.are_all_lower_policies_finished:
                    trainer.payoff_table_needs_update_started = True

            # refresh payoff table/selection probs
            if trainer.payoff_table_needs_update_started and trainer.new_payoff_table_promise is None:

                trainer.new_payoff_table_promise = trainer.live_table_tracker.get_live_payoff_table_dill_pickled.remote(first_wait_for_n_seconds=2)
                trainer.payoff_table_needs_update_started = False

            if trainer.new_payoff_table_promise is not None:
                _process_new_live_payoff_table_result_if_ready(
                    trainer=trainer,
                    block_until_result_is_ready=not PIPELINE_LIVE_PAYOFF_TABLE_CALC_IS_ASYNCHRONOUS)


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
        #             print(f"sampled policy {selected_policy_spec.key} (loading weights)")
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
        #         else:
        #             print(f"sampled policy {selected_policy_spec.key} (weights already loaded)")
        #
        #     trainer.workers.foreach_worker(worker_sample_new_static_policy)

        def sample_new_static_policy_weights_for_each_worker_on_episode_start(params):
            policies = params['policy']

            static_policy = policies[STATIC_POLICY]

            if not hasattr(static_policy, 'static_policy_selection_probs'):
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

            if trainer.are_all_lower_policies_finished:

                should_submit = False
                submit_reason = None

                if not hasattr(trainer, 'previous_threshold_check_reward'):
                    trainer.previous_threshold_check_reward = -100.0
                    trainer.next_threshold_check_steps = SUBMISSION_THRESHOLD_STEPS_START + result['timesteps_total']

                    print(colored(f"Policy {trainer.claimed_policy_num}: first threshold check at {trainer.next_threshold_check_steps} steps", "white"))

                if SUBMISSION_IMPROVEMENT_THRESHOLD_PER_STEPS is not None and \
                    SUBMISSION_IMPROVEMENT_THRESHOLD_REWARD is not None:

                    if result['timesteps_total'] >= trainer.next_threshold_check_steps:
                        trainer.next_threshold_check_steps = max(
                            trainer.next_threshold_check_steps + SUBMISSION_IMPROVEMENT_THRESHOLD_PER_STEPS,
                            result['timesteps_total'] + 1)

                        target_reward = trainer.previous_threshold_check_reward + SUBMISSION_IMPROVEMENT_THRESHOLD_REWARD
                        result['target_reward'] = target_reward
                        measured_reward = result['policy_reward_mean'][TRAIN_POLICY]
                        print(colored(f"Policy {trainer.claimed_policy_num}: {result['timesteps_total']} steps: {TRAIN_POLICY} reward: {measured_reward}, target reward: {target_reward}", "white"))

                        if measured_reward < target_reward and \
                                (SUBMISSION_MIN_STEPS is None or result['timesteps_total'] >= SUBMISSION_MIN_STEPS):
                            should_submit = True
                            submit_reason = f"plateaued at {measured_reward} reward"
                            print(
                                colored(f"Policy {trainer.claimed_policy_num}: {result['timesteps_total']} steps: {TRAIN_POLICY} didn\'t reach target reward. Submitting policy.", "white"))
                        else:
                            print(colored(f"Policy {trainer.claimed_policy_num}: next threshold check at {trainer.next_threshold_check_steps} steps", "white"))

                        trainer.previous_threshold_check_reward = measured_reward

                if SUBMISSION_MAX_STEPS is not None and result['timesteps_total'] >= SUBMISSION_MAX_STEPS:
                    should_submit = True
                    submit_reason = f"hit max steps of {SUBMISSION_MAX_STEPS}"
                    print(colored(f"Policy {trainer.claimed_policy_num}: Trainer hit max steps. Submitting policy.", "white"))

                if should_submit:
                    assert submit_reason is not None
                    result['stop_signal'] = True
                    local_train_policy = trainer.workers.local_worker().policy_map[TRAIN_POLICY]

                    tags = [*SUBMISSION_POLICY_TAGS,
                            submit_reason,
                            f"timesteps: {result['timesteps_total']}",
                            f"episodes: {result['episodes_total']}",
                            f"iter: {result['training_iteration']}"]
                    if hasattr(local_train_policy, "init_tag"):
                        tags += local_train_policy.init_tag

                    checkpoints_dir = os.path.join(experiment_save_dir, "policy_submissions")
                    checkpoint_name = f"policy_{trainer.claimed_policy_num}_{datetime_str()}_iter_{result['training_iteration']}.dill"
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

                    ray_get_and_free(trainer.live_table_tracker.set_latest_key_for_claimed_policy.remote(new_key=policy_key))
                    ray_get_and_free(trainer.live_table_tracker.set_claimed_policy_as_finished.remote())
                    # trainer.live_table_tracker.exit_actor()

        def all_on_train_result_callbacks(params):
            checkpoint_and_set_static_policy_distribution_on_train_result_callback(params=params)
            stop_and_submit_if_not_improving_on_train_result_callback(params=params)
            # sample_new_static_policy_weights_for_each_worker_on_train_result_callback(params=params)

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
                raise ValueError(f"train_policy_mapping_fn: wasn't expecting an agent id of {agent_id}")

        def copy_with_noise(weights_dict):
            weights_wnoise_dict = {}
            noise_dict = {}
            for var_name, w in weights_dict.items():
                if var_name.startswith("pi"):  # Ensure only modify policy, not value
                    noise_dict[var_name] = np.random.randn(*w.shape)
                    weights_wnoise_dict[var_name] = w.copy() + EVO_SIGMA*noise_dict[var_name]
                else:
                    weights_wnoise_dict[var_name] = w.copy()
            return weights_wnoise_dict, noise_dict

        def update_weights_to_fitness(weights_dict, evo_noise, evo_fitness_scores):
            weights_fitness_dict = copy.deepcopy(weights_dict)
            for e_j, f_j in zip(evo_noise, evo_fitness_scores):
                for var_name, var in weights_dict.items():
                    if var_name.startswith("pi"):  # Ensure only modify policy, not value
                        weights_fitness_dict[var_name] += f_j*(EVO_LR / (EVO_WKS*EVO_SIGMA))*e_j[var_name]
            return weights_fitness_dict

        def save_weights(trainer, weights_dict, dir_name, name, iter_num):
            # experiment_save_dir = /home/ubuntu/ray_results/leduc_poker_pipe_1_workers_learner_leduc_poker_sac_arch1_pipeline_psro_Area-51-16_pid_27214_04.08.09PM_Nov-25-2020
            checkpoints_dir = os.path.join(experiment_save_dir, dir_name)  # New dir for this
            checkpoint_name = f"policy_{trainer.claimed_policy_num}_{datetime_str()}_{name}_{iter_num}.dill"
            checkpoint_save_path = os.path.join(checkpoints_dir, checkpoint_name)
            ensure_dir(checkpoint_save_path)
            with open(checkpoint_save_path, "wb") as dill_file:
                dump(obj=weights_dict, file=dill_file)
            return checkpoint_save_path

        def fitness_score(payoff_array):
            n = payoff_array.shape[0]
            norm = np.linalg.norm(payoff_array, axis=1)
            payoff_array = payoff_array / norm[:, np.newaxis]
            L = payoff_array@payoff_array.T
            L_card = np.trace(np.eye(n)-np.linalg.inv(L+np.eye(n)))
            return L_card


        def evo_update(params):
            print(f"{colored(f'#####################################', 'magenta')}\n")
            print(f"{colored(f'############ EVO UPDATE #############', 'magenta')}\n")
            print(f"{colored(f'#####################################', 'magenta')}\n")
            time.sleep(3600)
            trainer = params['trainer']
            result = params['result']

            if trainer.my_weights_updates is None:
                trainer.my_weights_updates = 0

            if not trainer.my_weights_updates%EVO_UPDATE_FREQ==0:
                trainer.my_weights_updates += 1
                return
            trainer.my_weights_updates += 1

            # Add noise to weights
            time.sleep(1)
            local_train_policy = trainer.workers.local_worker().policy_map[TRAIN_POLICY]
            weights_dict = local_train_policy.get_model_weights(remove_scope_prefix=TRAIN_POLICY)
            evo_weights, evo_noise, evo_fitness_scores = [], [], []
            for j in range(EVO_WKS):
                # Get weights with noise
                weights_wnoise_dict, noise_dict = copy_with_noise(weights_dict)
                evo_weights.append(weights_wnoise_dict)
                evo_noise.append(noise_dict)

                checkpoint_save_path = save_weights(trainer, weights_wnoise_dict, "EVO_policy_checkpoints", "evowks", j)
                my_key = checkpoint_save_path

                # print(f"{colored(f'############ {my_key} #############', 'magenta')}\n")

                # Get payoff table for given EVO_WKS
                new_payoff_table_EVO_array = \
                    ray_get_and_free(trainer.live_table_tracker.my_get_live_payoff_table_dill_pickled\
                        .remote(new_weight_key=my_key, first_wait_for_n_seconds=0.1))

                # Compute fitness score
                if new_payoff_table_EVO_array is None:
                    print(f"{colored(f'############ No payoff table #############', 'magenta')}\n")
                    return

                # print(new_payoff_table_EVO_array.shape)
                evo_fitness_scores.append(fitness_score(new_payoff_table_EVO_array))

            # Update weights
            print(colored(evo_fitness_scores, 'magenta'))
            weights_fitness_dict = update_weights_to_fitness(weights_dict, evo_noise, evo_fitness_scores)

            # Set weights and compute fitness
            def worker_set_train_policy_weights2(worker):
                train_policy = worker.policy_map[TRAIN_POLICY]
                train_policy.set_model_weights(weights=weights_fitness_dict,
                                               add_scope_prefix=TRAIN_POLICY)
            trainer.workers.foreach_worker(worker_set_train_policy_weights2)

            _do_live_policy_checkpoint(trainer, result['training_iteration']+1)

            # Save Checkpoint fit weights
            checkpoint_save_path = save_weights(trainer, weights_fitness_dict,
                                                "EVO_FINAL_policy_checkpoints", "iter", result['training_iteration']+1)
            my_key = checkpoint_save_path

            new_payoff_table_EVO = \
                ray_get_and_free(trainer.live_table_tracker.my_get_live_payoff_table_dill_pickled \
                                 .remote(new_weight_key=my_key, first_wait_for_n_seconds=0.1))

            # Compute fitness
            final_fitness = fitness_score(new_payoff_table_EVO)  # TODO: save this somewhere
            print('Final fitness and stats')
            # print(f"{colored(f'############ CHECK ! #############', 'magenta')}\n")
            # time.sleep(3600)
            return


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
                claim_new_active_policy_after_trainer_init_callback,
                # evo_update,
            ],
            # "callbacks_after_optim_step": [
            #     evo_update,
            # ],
            "callbacks": {
                "on_train_result": all_on_train_result_callbacks,
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
            upload_dir=base_experiment_name,
            sync_to_cloud=get_tune_sync_to_cloud_fn(storage_client=storage_client, bucket_name=BUCKET_NAME),
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
