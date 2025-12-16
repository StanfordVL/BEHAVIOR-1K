import pymxs

rt = pymxs.runtime


def spherify(obj):
    if rt.classOf(obj) == rt.VolumeHelper:
        assert obj.volumeType == 1, f"{obj.name} is not a sphere."
    s = rt.Sphere()
    s.position = obj.position
    s.name = obj.name
    s.radius = obj.size / 2.0 if rt.classOf(obj) == rt.VolumeHelper else 20
    s.layer = obj.layer
    s.parent = obj.parent
    rt.delete(obj)
    return s


def main():
    rt.select([spherify(x) for x in rt.selection])


if __name__ == "__main__":
    main()
