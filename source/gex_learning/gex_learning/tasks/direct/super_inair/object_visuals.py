"""Rendering fixes for imported object meshes, without changing their physics."""


def make_visual_materials_opaque(object_prim):
    """Override the tennis OBJ's Tr=1 material in the scene, retaining its texture.

    URDF import instances the visual subtree even with make_instanceable=False.
    Expand only visual instances so shader overrides can be authored locally;
    the source asset and collision subtree remain untouched.
    """
    from pxr import Usd

    count = 0
    visual_roots = [prim for prim in Usd.PrimRange(object_prim) if prim.GetName() == "visuals"]
    for root in visual_roots:
        pending = [root]
        while pending:
            prim = pending.pop()
            if prim.IsInstance():
                prim.SetInstanceable(False)
            opacity = prim.GetAttribute("inputs:opacity_constant")
            if opacity.IsValid():
                opacity.Set(1.0)
                enabled = prim.GetAttribute("inputs:enable_opacity")
                if enabled.IsValid():
                    enabled.Set(False)
                count += 1
            pending.extend(prim.GetChildren())
    return count
