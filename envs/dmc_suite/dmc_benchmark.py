DOMAINS = [
    'walker',
    'quadruped',
    'jaco',
    'hopper',
    'cheetah'
]

WALKER_TASKS = [
    'walker_stand',
    'walker_walk',
    'walker_run',
    'walker_flip',
]

HOPPER_TASKS = [
    'hopper_stand',
    'hopper_hop',
    'hopper_hop_backward',
    'hopper_flip',
    'hopper_flip_backward'
]

CHEETAH_TASKS = [
    'cheetah_run',
    'cheetah_run_backward',
    'cheetah_flip',
    'cheetah_flip_backward'
]

QUADRUPED_TASKS = [
    'quadruped_walk',
    'quadruped_run',
    'quadruped_stand',
    'quadruped_jump',
]

JACO_TASKS = [
    'jaco_reach_top_left',
    'jaco_reach_top_right',
    'jaco_reach_bottom_left',
    'jaco_reach_bottom_right',
]

TASKS = WALKER_TASKS + QUADRUPED_TASKS + JACO_TASKS + CHEETAH_TASKS + HOPPER_TASKS

PRIMAL_TASKS = {
    'walker': 'walker_stand',
    'jaco': 'jaco_reach_top_left',
    'quadruped': 'quadruped_walk',
    'hopper': 'hopper_stand',
    'cheetah': 'cheetah_run'
}
