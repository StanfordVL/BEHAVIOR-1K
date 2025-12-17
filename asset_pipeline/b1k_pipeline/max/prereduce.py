import pymxs

rt = pymxs.runtime

VERTEX_COUNT = 20000

STOP_AFTER_EVERY_OBJECT = True


def prereduce():
    done_bases = set()
    for obj in rt.objects:
        # Once per set of instances.
        if obj.baseObject in done_bases:
            continue

        # Only for complex objects.
        if len(obj.vertices) <= VERTEX_COUNT:
            continue

        # Only do objects that don't already have multires on them.
        if any(
            [
                rt.classOf(x) == rt.multiRes or rt.classOf(x) == rt.ProOptimizer
                for x in obj.modifiers
            ]
        ):
            continue

        print(f"Processing object {obj.name}.")

        # multires = rt.multiRes()
        # rt.addModifier(obj, multires)

        # rt.windows.ProcessPostedMessages()

        # multires.reqGenerate = True

        # # For some reason this needs to be called twice
        # rt.windows.ProcessPostedMessages()

        # multires.vertexCount = VERTEX_COUNT

        rt.select(obj)

        optimizer = rt.ProOptimizer()
        rt.addModifier(obj, optimizer)
        optimizer.KeepTextures = True
        optimizer.Calculate = True

        # Nasty hack to force Calculate to be executed
        rt.completeRedraw()

        optimizer.vertexCount = VERTEX_COUNT

        edit_poly = rt.edit_poly()
        rt.addModifier(obj, edit_poly)

        if STOP_AFTER_EVERY_OBJECT:
            rt.select(obj)
            rt.IsolateSelection.EnterIsolateSelectionMode()
            return

        done_bases.add(obj.baseObject)

    print("Prereduce done.")
    rt.IsolateSelection.ExitIsolateSelectionMode()
    if not STOP_AFTER_EVERY_OBJECT:
        rt.messageBox(
            f"{len(done_bases)} objects processed. Please confirm visually reasonable vertex count (and reduce as necessary)."
        )


if __name__ == "__main__":
    prereduce()
