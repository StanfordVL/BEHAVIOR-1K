from pymxs import runtime as rt


def triangulate(obj):
    # Triangulate
    ttp = rt.Turn_To_Poly()
    ttp.limitPolySize = True
    ttp.maxPolySize = 3
    rt.addmodifier(obj, ttp)
    rt.maxOps.collapseNodeTo(obj, 1, True)


def main():
    for obj in rt.selection:
        triangulate(obj)


if __name__ == "__main__":
    main()
