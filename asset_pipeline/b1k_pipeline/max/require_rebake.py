from pymxs import runtime as rt


def require_rebake(obj):
    if not obj.material or rt.classOf(obj.material) != rt.Shell_Material:
        return

    # Switch the material to the shell's original material
    obj.material = obj.material.originalMaterial

    print("Successfully removed baked material from", obj.name)


def main():
    for obj in rt.selection:
        require_rebake(obj)


if __name__ == "__main__":
    main()
