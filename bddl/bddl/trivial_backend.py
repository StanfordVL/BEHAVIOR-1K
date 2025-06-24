from bddl.logic_base import UnaryAtomicFormula, BinaryAtomicFormula
from bddl.backend_abc import BDDLBackend
from bddl.parsing import parse_domain


*__, domain_predicates = parse_domain("omnigibson")
UNARIES = [predicate for predicate, inputs in domain_predicates.items() if len(inputs) == 1]
BINARIES = [predicate for predicate, inputs in domain_predicates.items() if len(inputs) == 2]


class TrivialBackend(BDDLBackend):
    def get_predicate_class(self, predicate_name):
        PREDICATE_MAPPING = {
            "cooked": TrivialCookedPredicate,
            "frozen": TrivialFrozenPredicate,
            "open": TrivialOpenPredicate,
            "folded": TrivialFoldedPredicate,
            "unfolded": TrivialUnfoldedPredicate,
            "toggled_on": TrivialToggledOnPredicate,
            "hot": TrivialHotPredicate,
            "frozen": TrivialFrozenPredicate,
            "on_fire": TrivialOnFirePredicate,
            "empty": TrivialEmptyPredicate,
            "closed": TrivialClosedPredicate,
            "future": TrivialFuturePredicate,
            "real": TrivialRealPredicate,
            "covered": TrivialCoveredPredicate,
            "ontop": TrivialOnTopPredicate,
            "inside": TrivialInsidePredicate,
            "filled": TrivialFilledPredicate,
            "saturated": TrivialSaturatedPredicate,
            "contains": TrivialContainsPredicate,
            "ontop": TrivialOnTopPredicate,
            "nextto": TrivialNextToPredicate,
            "under": TrivialUnderPredicate,
            "touching": TrivialTouchingPredicate,
            "overlaid": TrivialOverlaidPredicate,
            "attached": TrivialAttachedPredicate,
            "draped": TrivialDrapedPredicate,
            "insource": TrivialInsourcePredicate,
            "broken": TrivialBrokenPredicate,
            "grasped": TrivialGraspedPredicate,
        } 
        return PREDICATE_MAPPING[predicate_name]


class TrivialSimulator(object):
    def __init__(self):
        # Unaries - populated with 1-tuples of string names 
        self.cooked = set()
        self.frozen = set()
        self.open = set()
        self.folded = set()
        self.unfolded = set()
        self.toggled_on = set() 
        self.hot = set() 
        self.on_fire = set() 
        self.empty = set() 
        self.future = set() 
        self.real = set() 
        self.broken = set()
        self.closed = set()
        # Binaries - populated with 2-tuples of string names
        self.saturated = set()
        self.covered = set() 
        self.filled = set() 
        self.contains = set() 
        self.ontop = set() 
        self.nextto = set() 
        self.under = set() 
        self.touching = set() 
        self.inside = set() 
        self.overlaid = set() 
        self.attached = set() 
        self.draped = set() 
        self.insource = set() 
        self.grasped = set()

        self.create_predicate_to_setters()
    
    def create_predicate_to_setters(self):
        self.predicate_to_setters = {
            "cooked": self.set_cooked,
            "frozen": self.set_frozen,
            "open": self.set_open,
            "folded": self.set_folded,
            "unfolded": self.set_unfolded,
            "toggled_on": self.set_toggled_on,
            "hot": self.set_hot,
            "on_fire": self.set_on_fire,
            "empty": self.set_empty,
            "broken": self.set_broken,
            "closed": self.set_closed,
            "future": self.set_future,
            "real": self.set_real,
            "inside": self.set_inside,
            "ontop": self.set_ontop,
            "covered": self.set_covered,
            "filled": self.set_filled,
            "saturated": self.set_saturated,
            "nextto": self.set_nextto,
            "contains": self.set_contains,
            "under": self.set_under,
            "touching": self.set_touching,
            "overlaid": self.set_overlaid,
            "attached": self.set_attached,
            "draped": self.set_draped,
            "insource": self.set_insource,
            "grasped": self.set_grasped,
        }

    def set_state(self, literals): 
        """
        Given a set of non-contradictory parsed ground literals, set this backend to them. 
        Also set implied predicates:
            filled => contains
            not contains => not filled 
            ontop => nextto
            not nextto => not ontop
            TODO under? others? draped?
        """
        for literal in literals: 
            is_predicate = not(literal[0] == "not")
            predicate, *objects = literal[1] if (literal[0] == "not") else literal
            if predicate == "inroom": 
                print(f"Skipping inroom literal {literal}")
                continue
            self.predicate_to_setters[predicate](tuple(objects), is_predicate)
            # Entailed predicates 
            if is_predicate and (predicate == "filled"):
                self.predicate_to_setters["contains"](tuple(objects), True)
            if (not is_predicate) and (predicate == "contains"):
                self.predicate_to_setters["filled"](tuple(objects), False)
            if is_predicate and (predicate == "ontop"):
                self.predicate_to_setters["nextto"](tuple(objects), True)
            if (not is_predicate) and (predicate == "nextto"):
                self.predicate_to_setters["ontop"](tuple(objects), False)

    def set_cooked(self, objs, is_cooked):
        assert len(objs) == 1, f"`objs` has len other than 1: {objs}"
        if is_cooked: 
            self.cooked.add(objs)
        else: 
            self.cooked.discard(objs)
    
    def get_cooked(self, objs):
        return tuple(obj.name for obj in objs) in self.cooked
    
    def set_frozen(self, objs, is_frozen):
        assert len(objs) == 1, f"`objs` has len other than 1: {objs}"
        if is_frozen: 
            self.frozen.add(objs)
        else: 
            self.frozen.discard(objs)
    
    def get_frozen(self, objs):
        return tuple(obj.name for obj in objs) in self.frozen
    
    def set_open(self, objs, is_open):
        assert len(objs) == 1, f"`objs` has len other than 1: {objs}"
        if is_open: 
            self.open.add(objs)
        else: 
            self.open.discard(objs)
    
    def get_open(self, objs):
        return tuple(obj.name for obj in objs) in self.open
    
    def set_folded(self, objs, is_folded):
        assert len(objs) == 1, f"`objs` has len other than 1: {objs}"
        if is_folded: 
            self.folded.add(objs)
        else: 
            self.folded.discard(objs)
    
    def get_folded(self, objs):
        return tuple(obj.name for obj in objs) in self.folded
    
    def set_unfolded(self, objs, is_unfolded):
        assert len(objs) == 1, f"`objs` has len other than 1: {objs}"
        if is_unfolded: 
            self.unfolded.add(objs)
        else: 
            self.unfolded.discard(objs)
    
    def get_unfolded(self, objs):
        return tuple(obj.name for obj in objs) in self.unfolded
    
    def set_toggled_on(self, objs, is_toggled_on):
        assert len(objs) == 1, f"`objs` has len other than 1: {objs}"
        if is_toggled_on: 
            self.toggled_on.add(objs)
        else: 
            self.toggled_on.discard(objs)
    
    def get_toggled_on(self, objs):
        return tuple(obj.name for obj in objs) in self.toggled_on
    
    def set_hot(self, objs, is_hot):
        assert len(objs) == 1, f"`objs` has len other than 1: {objs}"
        if is_hot: 
            self.hot.add(objs)
        else: 
            self.hot.discard(objs)
    
    def get_hot(self, objs):
        return tuple(obj.name for obj in objs) in self.hot
    
    def set_on_fire(self, objs, is_on_fire):
        assert len(objs) == 1, f"`objs` has len other than 1: {objs}"
        if is_on_fire: 
            self.on_fire.add(objs)
        else: 
            self.on_fire.discard(objs)
    
    def get_on_fire(self, objs):
        return tuple(obj.name for obj in objs) in self.on_fire
    
    def set_empty(self, objs, is_empty):
        assert len(objs) == 1, f"`objs` has len other than 1: {objs}"
        if is_empty: 
            self.empty.add(objs)
        else: 
            self.empty.discard(objs)
    
    def get_empty(self, objs):
        return tuple(obj.name for obj in objs) in self.empty
    
    def set_closed(self, objs, is_closed):
        assert len(objs) == 1, f"`objs` has len other than 1: {objs}"
        if is_closed: 
            self.closed.add(objs)
        else: 
            self.closed.discard(objs)
    
    def get_closed(self, objs):
        return tuple(obj.name for obj in objs) in self.closed
    
    def set_broken(self, objs, is_broken):
        assert len(objs) == 1, f"`objs` has len other than 1: {objs}"
        if is_broken: 
            self.broken.add(objs)
        else: 
            self.broken.discard(objs)
    
    def get_broken(self, objs):
        return tuple(obj.name for obj in objs) in self.broken
    
    def set_future(self, objs, is_future):
        assert len(objs) == 1, f"`objs` has len other than 1: {objs}"
        if is_future: 
            self.future.add(objs)
        else: 
            self.future.discard(objs)
    
    def get_future(self, objs):
        return tuple(obj.name for obj in objs) in self.future
    
    def set_real(self, objs, is_real):
        assert len(objs) == 1, f"`objs` has len other than 1: {objs}"
        if is_real: 
            self.real.add(objs)
        else: 
            self.real.discard(objs)
    
    def get_real(self, objs):
        return tuple(obj.name for obj in objs) in self.real

    def set_covered(self, objs, is_covered):
        assert len(objs) == 2, f"`objs` has length other than 2: {objs}"
        if is_covered:
            self.covered.add(objs)
        else:
            self.covered.discard(objs)

    def get_covered(self, objs):
        return tuple(obj.name for obj in objs) in self.covered
    
    def set_ontop(self, objs, is_ontop):
        assert len(objs) == 2, f"`objs` has length other than 2: {objs}"
        if is_ontop:
            self.ontop.add(objs)
        else:
            self.ontop.discard(objs)

    def get_ontop(self, objs):
        return tuple(obj.name for obj in objs) in self.ontop
    
    def set_inside(self, objs, is_inside):
        assert len(objs) == 2, f"`objs` has length other than 2: {objs}"
        if is_inside:
            self.inside.add(objs)
        else:
            self.inside.discard(objs)
    
    def get_inside(self, objs):
        return tuple(obj.name for obj in objs) in self.inside
    
    def set_filled(self, objs, is_filled):
        assert len(objs) == 2, f"`objs` has length other than 2: {objs}"
        if is_filled:
            self.filled.add(objs)
        else:
            self.filled.discard(objs)
    
    def get_filled(self, objs):
        return tuple(obj.name for obj in objs) in self.filled

    def set_saturated(self, objs, is_saturated):
        assert len(objs) == 2, f"`objs` has length other than 2: {objs}"
        if is_saturated:
            self.saturated.add(objs)
        else:
            self.saturated.discard(objs)
    
    def get_saturated(self, objs):
        return tuple(obj.name for obj in objs) in self.saturated

    def set_nextto(self, objs, is_nextto):
        assert len(objs) == 2, f"`objs` has length other than 2: {objs}"
        if is_nextto:
            self.nextto.add(objs)
        else:
            self.nextto.discard(objs)
    
    def get_nextto(self, objs):
        return tuple(obj.name for obj in objs) in self.nextto

    def set_contains(self, objs, is_contains):
        assert len(objs) == 2, f"`objs` has length other than 2: {objs}"
        if is_contains:
            self.contains.add(objs)
        else:
            self.contains.discard(objs)
    
    def get_contains(self, objs):
        return tuple(obj.name for obj in objs) in self.contains
    
    def set_under(self, objs, is_under):
        assert len(objs) == 2, f"`objs` has length other than 2: {objs}"
        if is_under:
            self.under.add(objs)
        else:
            self.under.discard(objs)
    
    def get_under(self, objs):
        return tuple(obj.name for obj in objs) in self.under

    def set_touching(self, objs, is_touching):
        assert len(objs) == 2, f"`objs` has length other than 2: {objs}"
        if is_touching:
            self.touching.add(objs)
        else:
            self.touching.discard(objs)
    
    def get_touching(self, objs):
        return tuple(obj.name for obj in objs) in self.touching
    
    def set_overlaid(self, objs, is_overlaid):
        assert len(objs) == 2, f"`objs` has length other than 2: {objs}"
        if is_overlaid:
            self.overlaid.add(objs)
        else:
            self.overlaid.discard(objs)
    
    def get_overlaid(self, objs):
        return tuple(obj.name for obj in objs) in self.overlaid
    
    def set_attached(self, objs, is_attached):
        assert len(objs) == 2, f"`objs` has length other than 2: {objs}"
        if is_attached:
            self.attached.add(objs)
        else:
            self.attached.discard(objs)
    
    def get_attached(self, objs):
        return tuple(obj.name for obj in objs) in self.attached

    def set_draped(self, objs, is_draped):
        assert len(objs) == 2, f"`objs` has length other than 2: {objs}"
        if is_draped:
            self.draped.add(objs)
        else:
            self.draped.discard(objs)
    
    def get_draped(self, objs):
        return tuple(obj.name for obj in objs) in self.draped
    
    def set_insource(self, objs, is_insource):
        assert len(objs) == 2, f"`objs` has length other than 2: {objs}"
        if is_insource:
            self.insource.add(objs)
        else:
            self.insource.discard(objs)
    
    def get_insource(self, objs):
        return tuple(obj.name for obj in objs) in self.insource
    
    def set_grasped(self, objs, is_grasped):
        assert len(objs) == 2, f"`objs` has length other than 2: {objs}"
        if is_grasped:
            self.grasped.add(objs)
        else:
            self.grasped.discard(objs)
    
    def get_grasped(self, objs):
        return tuple(obj.name for obj in objs) in self.grasped


class TrivialGenericObject(object): 
    def __init__(self, name, simulator):
        self.name = name
        self.simulator = simulator
    
    def get_cooked(self):
        return self.simulator.get_cooked((self,))
    
    def get_frozen(self):
        return self.simulator.get_frozen((self,))
    
    def get_open(self):
        return self.simulator.get_open((self,))
    
    def get_folded(self):
        return self.simulator.get_folded((self,))
    
    def get_unfolded(self):
        return self.simulator.get_unfolded((self,))
    
    def get_toggled_on(self):
        return self.simulator.get_toggled_on((self,))
    
    def get_closed(self):
        return self.simulator.get_closed((self,))
    
    def get_hot(self):
        return self.simulator.get_hot((self,))
    
    def get_on_fire(self):
        return self.simulator.get_on_fire((self,))
    
    def get_empty(self):
        return self.simulator.get_empty((self,))
    
    def get_broken(self):
        return self.simulator.get_broken((self,))
    
    def get_future(self):
        return self.simulator.get_future((self,))
    
    def get_real(self):
        return self.simulator.get_real((self,))
    
    def get_ontop(self, other):
        return self.simulator.get_ontop((self, other))
    
    def get_covered(self, other):
        return self.simulator.get_covered((self, other))

    def get_inside(self, other):
        return self.simulator.get_inside((self, other))
    
    def get_saturated(self, other):
        return self.simulator.get_saturated((self, other))

    def get_nextto(self, other):
        return self.simulator.get_nextto((self, other))

    def get_contains(self, other):
        return self.simulator.get_contains((self, other))

    def get_under(self, other):
        return self.simulator.get_under((self, other))

    def get_touching(self, other):
        return self.simulator.get_touching((self, other))

    def get_overlaid(self, other):
        return self.simulator.get_overlaid((self, other))

    def get_attached(self, other):
        return self.simulator.get_attached((self, other))

    def get_draped(self, other):
        return self.simulator.get_draped((self, other))

    def get_insource(self, other):
        return self.simulator.get_insource((self, other))
    
    def get_grasped(self, other):
        return self.simulator.get_grasped((self, other))


# OmniGibson trivial predicates
class TrivialCookedPredicate(UnaryAtomicFormula):
    STATE_NAME = "cooked"

    def _evaluate(self, obj):
        print(self.STATE_NAME, obj.name, obj.get_cooked())
        return obj.get_cooked()

    def _sample(self, obj1, binary_state):
        pass


class TrivialFrozenPredicate(UnaryAtomicFormula):
    STATE_NAME = "frozen"

    def _evaluate(self, obj):
        print(self.STATE_NAME, obj.name, obj.get_frozen())
        return obj.get_frozen()

    def _sample(self, obj1, binary_state):
        pass


class TrivialOpenPredicate(UnaryAtomicFormula):
    STATE_NAME = "open"

    def _evaluate(self, obj):
        print(self.STATE_NAME, obj.name, obj.get_open())
        return obj.get_open()

    def _sample(self, obj1, binary_state):
        pass


class TrivialFoldedPredicate(UnaryAtomicFormula):
    STATE_NAME = "folded"

    def _evaluate(self, obj):
        print(self.STATE_NAME, obj.name, obj.get_folded())
        return obj.get_folded()

    def _sample(self, obj1, binary_state):
        pass


class TrivialUnfoldedPredicate(UnaryAtomicFormula):
    STATE_NAME = "unfolded"

    def _evaluate(self, obj):
        print(self.STATE_NAME, obj.name, obj.get_unfolded())
        return obj.get_unfolded()

    def _sample(self, obj1, binary_state):
        pass


class TrivialToggledOnPredicate(UnaryAtomicFormula):
    STATE_NAME = "toggled_on"

    def _evaluate(self, obj):
        print(self.STATE_NAME, obj.name, obj.get_toggled_on())
        return obj.get_toggled_on()

    def _sample(self, obj1, binary_state):
        pass


class TrivialClosedPredicate(UnaryAtomicFormula):
    STATE_NAME = "closed"

    def _evaluate(self, obj):
        print(self.STATE_NAME, obj.name, obj.get_closed())
        return obj.get_closed()

    def _sample(self, obj1, binary_state):
        pass


class TrivialOnFirePredicate(UnaryAtomicFormula):
    STATE_NAME = "on_fire"

    def _evaluate(self, obj):
        print(self.STATE_NAME, obj.name, obj.get_on_fire())
        return obj.get_on_fire()

    def _sample(self, obj1, binary_state):
        pass


class TrivialHotPredicate(UnaryAtomicFormula):
    STATE_NAME = "hot"

    def _evaluate(self, obj):
        print(self.STATE_NAME, obj.name, obj.get_hot())
        return obj.get_hot()

    def _sample(self, obj1, binary_state):
        pass


class TrivialHotPredicate(UnaryAtomicFormula):
    STATE_NAME = "on_fire"

    def _evaluate(self, obj):
        print(self.STATE_NAME, obj.name, obj.get_on_fire())
        return obj.get_on_fire()

    def _sample(self, obj1, binary_state):
        pass


class TrivialEmptyPredicate(UnaryAtomicFormula):
    STATE_NAME = "on_fire"

    def _evaluate(self, obj):
        print(self.STATE_NAME, obj.name, obj.get_empty())
        return obj.get_empty()

    def _sample(self, obj1, binary_state):
        pass


class TrivialBrokenPredicate(UnaryAtomicFormula):
    STATE_NAME = "broken"

    def _evaluate(self, obj):
        print(self.STATE_NAME, obj.name, obj.get_broken())
        return obj.get_broken()

    def _sample(self, obj1, binary_state):
        pass


class TrivialFuturePredicate(UnaryAtomicFormula):
    STATE_NAME = "future"

    def _evaluate(self, obj):
        print(self.STATE_NAME, obj.name, obj.get_future())
        return obj.get_future()

    def _sample(self, obj1, binary_state):
        pass


class TrivialRealPredicate(UnaryAtomicFormula):
    STATE_NAME = "real"

    def _evaluate(self, obj):
        print(self.STATE_NAME, obj.name, obj.get_real())
        return obj.get_real()

    def _sample(self, obj1, binary_state):
        pass


class TrivialCoveredPredicate(BinaryAtomicFormula):
    STATE_NAME = "covered"

    def _evaluate(self, obj1, obj2):
        print(self.STATE_NAME, obj1.name, obj2.name, obj1.get_covered(obj2))
        return obj1.get_covered(obj2)

    def _sample(self, obj1, obj2, binary_state):
        pass


class TrivialInsidePredicate(BinaryAtomicFormula):
    STATE_NAME = "inside"

    def _evaluate(self, obj1, obj2):
        print(self.STATE_NAME, obj1.name, obj2.name, obj1.get_inside(obj2))
        return obj1.get_inside(obj2)

    def _sample(self, obj1, obj2, binary_state):
        pass


class TrivialOnTopPredicate(BinaryAtomicFormula):
    STATE_NAME = "ontop"

    def _evaluate(self, obj1, obj2):
        print(self.STATE_NAME, obj1.name, obj2.name, obj1.get_ontop(obj2))
        return obj1.get_ontop(obj2)

    def _sample(self, obj1, obj2, binary_state):
        pass


class TrivialFilledPredicate(BinaryAtomicFormula):
    STATE_NAME = "ontop"

    def _evaluate(self, obj1, obj2):
        print(self.STATE_NAME, obj1.name, obj2.name, obj1.get_ontop(obj2))
        return obj1.get_ontop(obj2)

    def _sample(self, obj1, obj2, binary_state):
        pass


class TrivialSaturatedPredicate(BinaryAtomicFormula):
    STATE_NAME = "saturated"

    def _evaluate(self, obj1, obj2):
        print(self.STATE_NAME, obj1.name, obj2.name, obj1.get_saturated(obj2))
        return obj1.get_saturated(obj2)

    def _sample(self, obj1, obj2, binary_state):
        pass


class TrivialNextToPredicate(BinaryAtomicFormula):
    STATE_NAME = "ontop"

    def _evaluate(self, obj1, obj2):
        print(self.STATE_NAME, obj1.name, obj2.name, obj1.get_nextto(obj2))
        return obj1.get_nextto(obj2)

    def _sample(self, obj1, obj2, binary_state):
        pass


class TrivialContainsPredicate(BinaryAtomicFormula):
    STATE_NAME = "ontop"

    def _evaluate(self, obj1, obj2):
        print(self.STATE_NAME, obj1.name, obj2.name, obj1.get_contains(obj2))
        return obj1.get_contains(obj2)

    def _sample(self, obj1, obj2, binary_state):
        pass


class TrivialUnderPredicate(BinaryAtomicFormula):
    STATE_NAME = "under"

    def _evaluate(self, obj1, obj2):
        print(self.STATE_NAME, obj1.name, obj2.name, obj1.get_under(obj2))
        return obj1.get_under(obj2)

    def _sample(self, obj1, obj2, binary_state):
        pass


class TrivialTouchingPredicate(BinaryAtomicFormula):
    STATE_NAME = "ontop"

    def _evaluate(self, obj1, obj2):
        print(self.STATE_NAME, obj1.name, obj2.name, obj1.get_touching(obj2))
        return obj1.get_touching(obj2)

    def _sample(self, obj1, obj2, binary_state):
        pass


class TrivialOverlaidPredicate(BinaryAtomicFormula):
    STATE_NAME = "ontop"

    def _evaluate(self, obj1, obj2):
        print(self.STATE_NAME, obj1.name, obj2.name, obj1.get_overlaid(obj2))
        return obj1.get_overlaid(obj2)

    def _sample(self, obj1, obj2, binary_state):
        pass


class TrivialAttachedPredicate(BinaryAtomicFormula):
    STATE_NAME = "ontop"

    def _evaluate(self, obj1, obj2):
        print(self.STATE_NAME, obj1.name, obj2.name, obj1.get_attached(obj2))
        return obj1.get_attached(obj2)

    def _sample(self, obj1, obj2, binary_state):
        pass


class TrivialDrapedPredicate(BinaryAtomicFormula):
    STATE_NAME = "draped"

    def _evaluate(self, obj1, obj2):
        print(self.STATE_NAME, obj1.name, obj2.name, obj1.get_draped(obj2))
        return obj1.get_draped(obj2)

    def _sample(self, obj1, obj2, binary_state):
        pass


class TrivialInsourcePredicate(BinaryAtomicFormula):
    STATE_NAME = "insource"

    def _evaluate(self, obj1, obj2):
        print(self.STATE_NAME, obj1.name, obj2.name, obj1.get_insource(obj2))
        return obj1.get_insource(obj2)

    def _sample(self, obj1, obj2, binary_state):
        pass


class TrivialGraspedPredicate(BinaryAtomicFormula):
    STATE_NAME = "grasped"

    def _evaluate(self, obj1, obj2):
        print(self.STATE_NAME, obj1.name, obj2.name, obj1.get_grasped(obj2))
        return obj1.get_grasped(obj2)

    def _sample(self, obj1, obj2, binary_state):
        pass


VALID_ATTACHMENTS = set([
    ("mixing_bowl.n.01", "electric_mixer.n.01"),
    ("cork.n.04", "wine_bottle.n.01"),
    ("menu.n.01", "wall.n.01"),
    ("broken__light_bulb.n.01", "table_lamp.n.01"),
    ("light_bulb.n.01", "table_lamp.n.01"),
    ("lens.n.01", "digital_camera.n.01"),
    ("screen.n.01", "wall.n.01"),
    ("antler.n.01", "wall.n.01"),
    ("skateboard_wheel.n.01", "skateboard.n.01"),
    ("blackberry.n.01", "scrub.n.01"),
    ("raspberry.n.02", "scrub.n.01"),
    ("dip.n.07", "candlestick.n.01"),
    ("sign.n.02", "wall.n.01"),
    ("wreath.n.01", "wall.n.01"),
    ("bow.n.08", "wall.n.01"),
    ("holly.n.03", "wall.n.01"),
    ("curtain_rod.n.01", "wall.n.01"),
    ("bicycle.n.01", "bicycle_rack.n.01"),
    ("bicycle_rack.n.01", "wall.n.01"),
    ("dartboard.n.01", "wall.n.01"),
    ("rug.n.01", "wall.n.01"),
    ("fairy_light.n.01", "wall.n.01"),
    ("lantern.n.01", "wall.n.01"),
    ("address.n.05", "wall.n.01"),
    ("hanger.n.02", "wardrobe.n.01"),
    ("flagpole.n.02", "wall.n.01"),
    ("picture_frame.n.01", "wall.n.01"),
    ("wind_chime.n.01", "pole.n.01"),
    ("pole.n.01", "wall.n.01"),
    ("hook.n.05", "trailer_truck.n.01"),
    ("fire_alarm.n.02", "wall.n.01"),
    ("poster.n.01", "wall.n.01"),
    ("painting.n.01", "wall.n.01"),
    ("hanger.n.02", "coatrack.n.01"),
    ("license_plate.n.01", "car.n.01"),
    ("gummed_label.n.01", "license_plate.n.01"),
    ("wallpaper.n.01", "wall.n.01"),
    ("mirror.n.01", "wall.n.01"),
    ("webcam.n.02", "desktop_computer.n.01"),
    ("kayak.n.01", "kayak_rack.n.01"),
    ("kayak_rack.n.01", "wall.n.01"),
    ("fish.n.02", "fishing_rod.n.01"),
    ("bicycle_rack.n.01", "recreational_vehicle.n.01"),
])

VALID_ROOMS = set()