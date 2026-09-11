"""FlashSAC: an off-policy alternative to PPO for mjlab velocity tasks.

Ported from the standalone ``rsl_rl_flashsac`` package
(https://github.com — see ``/home/miguel/fun/rsl_rl_flashsac``), which
implements FlashSAC directly against the ``rsl_rl`` algorithm/runner
interface. Only ``playground.rl.flashsac.runner`` and
``playground.rl.flashsac.config`` are new here, replacing the
Isaac-Lab-specific runner config and env wrapper from that project's sibling
``isaaclab_flashsac`` package with mjlab equivalents.
"""

from playground.rl.flashsac.config import (
  RslRlFlashSacActorCfg as RslRlFlashSacActorCfg,
)
from playground.rl.flashsac.config import (
  RslRlFlashSacAlgorithmCfg as RslRlFlashSacAlgorithmCfg,
)
from playground.rl.flashsac.config import (
  RslRlFlashSacCriticCfg as RslRlFlashSacCriticCfg,
)
from playground.rl.flashsac.config import (
  RslRlFlashSacRunnerCfg as RslRlFlashSacRunnerCfg,
)
from playground.rl.flashsac.runner import (
  MjlabOffPolicyRunner as MjlabOffPolicyRunner,
)
from playground.rl.flashsac.runner import (
  VelocityOffPolicyRunner as VelocityOffPolicyRunner,
)
