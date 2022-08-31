import random
from typing import List, Tuple


def rand_decision(p):
    return random.random() <= p


class NLPTransform:
    def __init__(self, transform, p=1.0) -> None:
        self.transform = transform
        self.p = p
        self.name = transform.name

    def __call__(self, text):
        if rand_decision(self.p):
            text = self.transform.augment(text)[0]
        return text

    def __repr__(self) -> str:
        return str(self.transform)

    def to_dict(self):
        return {"name": self.name, "transform": str(self.transform), "p": self.p}


class OneOfTransfroms:
    def __init__(self, transforms: List[NLPTransform], p=1.0, name="OneOf") -> None:
        self.transforms = transforms
        self.p = p
        self.name = name

    def __call__(self, text):
        if rand_decision(self.p):
            text = random.choice(self.transforms)(text)
        return text

    def __repr__(self) -> str:
        return "\n".join([str(t) for t in self.transforms])

    def to_dict(self):
        return {
            "name": self.name,
            "transforms": [t.to_dict() for t in self.transforms],
            "p": self.p,
        }


class ComposeAug:
    def __init__(
        self, transforms: Tuple[OneOfTransfroms, NLPTransform], p=1.0, name="Compose"
    ) -> None:
        self.transforms = transforms
        self.p = p
        self.name = name

    def __call__(self, text):
        if rand_decision(self.p):
            for t in self.transforms:
                text = t(text)
        return text

    def __repr__(self) -> str:
        return "\n".join([str(t) for t in self.transforms])

    def to_dict(self):
        return {
            "name": self.name,
            "transforms": [t.to_dict() for t in self.transforms],
            "p": self.p,
        }
