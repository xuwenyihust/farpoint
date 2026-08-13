# SO-101 recovery demonstrations

Recovery demonstrations begin from a live policy-induced deviation, not from a
normal scene reset and not from a rendered frame reconstructed after the fact.
An existing failed rollout may select the training scene, failure class, and
approximate intervention stage. Collection then reruns the frozen policy and
scene seed. When a versioned trigger fires, control passes to the Oracle without
resetting Isaac/PhysX state.

The recovery episode starts at that handoff. Frame zero records the measured
robot, object, contact, and simulation state; action zero is the first Oracle
correction. Policy actions before handoff remain in the source rollout evidence
and are not exported as recovery supervision.

`farpoint.demonstration.v1` distinguishes nominal and recovery episodes without
changing policy features. Recovery records bind the source checkpoint, trigger,
failure class, source control step, continuous-state handoff, state snapshot
hash, Oracle profile, and recovery strategy. Full traces remain in Farpoint
sidecars. LeRobot output continues to expose the same state, front/wrist image,
and action features.

Recovery demonstrations are training-only. Final rollout holdouts must never be
used as collection, intervention, replacement, or checkpoint-selection scenes.
