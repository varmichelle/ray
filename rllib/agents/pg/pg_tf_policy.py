"""
TensorFlow policy class used for PG.
"""

from typing import List, Type, Union
import numpy as np

import ray
from ray.rllib.agents.pg.utils import post_process_advantages
from ray.rllib.evaluation.postprocessing import Postprocessing, discount_cumsum
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy import Policy
from ray.rllib.policy.tf_policy_template import build_tf_policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.typing import TensorType

import sys
sys.path.insert(0, '~/Github/avoiding-cop')
import main

tf1, tf, tfv = try_import_tf()


def pg_tf_loss(
        policy: Policy, model: ModelV2, dist_class: Type[ActionDistribution],
        train_batch: SampleBatch) -> Union[TensorType, List[TensorType]]:
    """The basic policy gradients loss function.

    Args:
        policy (Policy): The Policy to calculate the loss for.
        model (ModelV2): The Model to calculate the loss for.
        dist_class (Type[ActionDistribution]: The action distr. class.
        train_batch (SampleBatch): The training data.

    Returns:
        Union[TensorType, List[TensorType]]: A single loss tensor or a list
            of loss tensors.
    """
    # Update advantages with power intrinsic reward 
    update_advantages_with_power(policy, train_batch)

    # Pass the training data through our model to get distribution parameters.
    dist_inputs, _ = model(train_batch)

    # Create an action distribution object.
    action_dist = dist_class(dist_inputs, model)

    # Calculate the vanilla PG loss based on:
    # L = -E[ log(pi(a|s)) * A]
    loss = -tf.reduce_mean(
        action_dist.logp(train_batch[SampleBatch.ACTIONS]) * tf.cast(
            train_batch[Postprocessing.ADVANTAGES], dtype=tf.float32))

    return loss


def update_advantages_with_power(policy: Policy, train_batch: SampleBatch):
    power_rewards = main.compute_power(train_batch)
    if power_rewards is None:
        return
    infos = train_batch[SampleBatch.INFOS]
    traj_len = int(infos[0,-1])
    traj_rewards_list = np.array_split(power_rewards, int(power_rewards.shape[0]/traj_len))
    processed_im_list = []
    for traj_rewards in traj_rewards_list:
        traj_rewards = np.concatenate([traj_rewards, np.array([0])])  # 0 for 0 value state at end of traj
        # print('traj_rewards', traj_rewards)
        processed_im_for_traj = discount_cumsum(traj_rewards, policy.config["gamma"])[:-1].astype(np.float32)
        # print('processed_im_for_traj', processed_im_for_traj)
        processed_im_list.append(processed_im_for_traj)
    final_power = np.concatenate(processed_im_list)
    train_batch[Postprocessing.ADVANTAGES] -= final_power


# Build a child class of `DynamicTFPolicy`, given the extra options:
# - trajectory post-processing function (to calculate advantages)
# - PG loss function
PGTFPolicy = build_tf_policy(
    name="PGTFPolicy",
    get_default_config=lambda: ray.rllib.agents.pg.pg.DEFAULT_CONFIG,
    postprocess_fn=post_process_advantages,
    loss_fn=pg_tf_loss)
