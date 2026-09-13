"""Phase 12/13: Desktop GUI and Rija Studio web playground."""

__all__ = ["RijaApp", "MyAIApp", "launch_app", "launch_studio"]


def __getattr__(name):
    if name in {"RijaApp", "MyAIApp", "launch_app"}:
        from ui.app import RijaApp, launch_app

        return {"RijaApp": RijaApp, "MyAIApp": RijaApp, "launch_app": launch_app}[name]
    if name == "launch_studio":
        from ui.studio import launch_studio

        return launch_studio
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
