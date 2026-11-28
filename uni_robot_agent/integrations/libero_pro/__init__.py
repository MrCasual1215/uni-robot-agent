"""LIBERO-PRO through the shared Direct gripper tools."""


def get_integration_spec():
    from uni_robot_agent.integrations.gripper import CLEAN_TOOLS, DIRECT_REACHABILITY_TOOLS, DIRECT_TOOLS
    from uni_robot_agent.integrations.simulation import build_simulation_spec
    # direct_reachability adds the read-only preview_move dry run over the adapter's kinematic pose plans.
    spec = build_simulation_spec("libero_pro", tool_profiles={"clean": CLEAN_TOOLS, "direct": DIRECT_TOOLS,
                                                             "direct_reachability": DIRECT_REACHABILITY_TOOLS})
    spec.deployment_module = "uni_robot_agent.integrations.libero_pro.deployment"
    return spec
