from mani_skill.agents.robots import UnitreeGo2, ANYmalC # imports your robot and registers it
# imports the demo_robot example script and lets you test your new robot
import mani_skill.examples.demo_robot as demo_robot_script
# TODO (stao): Anymal may not be modelled correctly or efficiently at the moment
import numpy as np
import sapien
import torch

from mani_skill import ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.utils import sapien_utils
from mani_skill.utils.structs.articulation import Articulation


@register_agent()
class PizzaRobot(BaseAgent):
    uid = "pizza_robot"
    urdf_path = f"./panda.urdf"

    fix_root_link = False
    disable_self_collisions = True

    # from panda
    keyframes = dict(
        rest=Keyframe(
            qpos=np.array(
                [
                    0.0,
                    0.0,
                    0.0,
                    np.pi / 2,
                    np.pi / 2,
                    -np.pi / 2,
                    np.pi / 2,
                    np.pi / 2,
                    0.0,
                ]
            ),
            pose=sapien.Pose(),
        )
    )

    joint_names = ['left_wheel_joint', 'right_wheel_joint', 
                   'panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 
                   'panda_joint5', 'panda_joint6', 'panda_joint7']

    arm_stiffness = 1e3
    arm_damping = 1e2
    arm_force_limit = 100

    @property
    def _controller_configs(self):
        # print([i.name for i in self.robot.active_joints])
        # delta action scale for Omni Isaac Gym Envs is self.dt * self.action_scale = 1/60 * 13.5. NOTE that their self.dt value is not the same as the actual DT used in sim...., they use default of 1/100
        pd_joint_delta_pos = PDJointPosControllerConfig(
            self.joint_names,
            -0.225,
            0.225,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            normalize_action=True,
            use_delta=True,
        )
        pd_joint_pos = PDJointPosControllerConfig(
            self.joint_names,
            None,
            None,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            normalize_action=False,
            use_delta=False,
        )
        # TODO (stao): For quadrupeds perhaps we disable gravit for all links except the root?
        controller_configs = dict(
            pd_joint_delta_pos=dict(
                body=pd_joint_delta_pos, balance_passive_force=False
            ),
            pd_joint_pos=dict(body=pd_joint_pos, balance_passive_force=False),
        )
        return controller_configs

    def _after_init(self):
        # disable gravity / compensate gravity automatically in all links but the root one
        for link in self.robot.links[1:]:
            link.disable_gravity = True

if __name__ == "__main__":
    demo_robot_script.main()
