# Phone/gRPC plugin

The custom `phone_grpc` teleoperator is maintained in the dedicated sibling
package `packages/lerobot-teleoperator-phone-grpc`. LeRobot loads it through
standard third-party plugin discovery; its built-in `phone` type remains upstream.

From this LeRobot workspace, install the local checkout with IK support:

```bash
uv pip install -e "../packages/lerobot-teleoperator-phone-grpc[ik]"
uv run --no-sync phone-grpc-server --help
```

Use `--teleop.type=phone_grpc` with `lerobot-teleoperate` or `lerobot-record`.
Supply `--teleop.urdf_path` and, when needed, `--teleop.mesh_path` for your robot.
The old checkout-relative SO100 model lookup is removed. Configure the matching
follower with `--robot.use_degrees=true`.

The plugin README documents the six-joint model convention, optional visualization,
gRPC receiver, IK backend, and development tests. The service and message field
identities used by the phone app are preserved. Install the plugin separately in
each runtime environment; it has not been added to LeRobot's dependency lockfile.
