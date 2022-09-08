import random
from typing import List, Tuple


def rand_decision(p: float) -> bool:
    return random.random() <= p


class NLPTransform:
    def __init__(self, transform, p: float = 1.0) -> None:
        """Wrapper for nlpaug transforms

        Parameters
        ----------
        transform : nlpaug transform
        p : float, optional
            probability for this transform, by default 1.0
        """
        self.transform = transform
        self.p = p
        self.name = transform.name

    def __call__(self, text):
        if rand_decision(self.p):
            text = self.transform.augment(text)[0]
        return text

    def __repr__(self) -> str:
        return str(self.transform)

    def to_dict(self) -> dict:
        """Return dict representation of this transfroms. Needed for logging

        Returns
        -------
        dict
        """
        return {"name": self.name, "transform": str(self.transform), "p": self.p}


class OneOfTransfroms:
    def __init__(
        self, transforms: List[NLPTransform], p: float = 1.0, name: str = "OneOf"
    ) -> None:
        """Implementation OneOf function from albumentations for nlpaug.
        Choose one of transform using uniform distribution

        Parameters
        ----------
        transforms : List[NLPTransform]
        p : float, optional
            probability for this transform, by default 1.0
        name : str, optional
            name for logging, by default "OneOf"
        """
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
        self,
        transforms: Tuple[OneOfTransfroms, NLPTransform],
        p: int = 1.0,
        name: str = "Compose",
    ) -> None:
        """Implementation Compose function from albumentations for nlpaug.
        Combines different transforms in one pipeline and execute them one by one

        Parameters
        ----------
        transforms : Tuple[OneOfTransfroms, NLPTransform]
        p : int, optional
            by default 1.0
        name : str, optional
            name for logging, by default "Compose"
        """
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
