# Sourccey plugin for LeRobot

This package registers the Sourccey robots and teleoperators with LeRobot.

For local development from the LeRobot repository root:

```bash
uv pip install --no-deps -e packages/lerobot_robot_sourccey
```

LeRobot commands discover the installed package automatically. For example:

```bash
lerobot-teleoperate --robot.type=sourccey --teleop.type=bi_sourccey_leader
```

Available robot types are `sourccey`, `sourccey_client`, and
`sourccey_follower`. Available teleoperator types are `sourccey_leader` and
`bi_sourccey_leader`.
