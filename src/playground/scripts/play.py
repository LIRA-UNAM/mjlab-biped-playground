"""Thin wrapper around mjlab's play script that defaults to the Viser viewer,
for its built-in interactive velocity-command joystick (mjlab.tasks.velocity.mdp
.UniformVelocityCommand.create_gui). Pass --viewer native to opt back into the
native window.
"""

import sys

import mjlab
import mjlab.tasks  # noqa: F401  (populates the task registry)
import tyro
from mjlab.scripts._cli import maybe_print_top_level_help
from mjlab.scripts.play import PlayConfig, run_play
from mjlab.tasks.registry import list_tasks, load_rl_cfg


def main():
  maybe_print_top_level_help("play")

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  agent_cfg = load_rl_cfg(chosen_task)
  args = tyro.cli(
    PlayConfig,
    args=remaining_args,
    default=PlayConfig(viewer="viser"),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  del remaining_args, agent_cfg

  run_play(chosen_task, args)


if __name__ == "__main__":
  main()
