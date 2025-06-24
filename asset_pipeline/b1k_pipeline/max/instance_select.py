import pymxs

rt = pymxs.runtime


def instance_select():
    roots = set(x.baseObject for x in rt.selection)
    objs = [x for x in rt.objects if x.baseObject in roots]
    rt.select(objs)


if __name__ == "__main__":
    instance_select()
