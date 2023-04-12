from collections import namedtuple
from typing import Optional

ActionXY = namedtuple('ActionXY', ['vx', 'vy'])
ActionRot = namedtuple('ActionRot', ['v', 'r'])


class ActionPoint:

    def __init__(self,
                 s_lin: Optional[float] = None,
                 s_ang: Optional[float] = None,
                 px: Optional[float] = None,
                 py: Optional[float] = None,
                 theta: Optional[float] = None,
                 vx: Optional[float] = None,
                 vy: Optional[float] = None,
                 omega: Optional[float] = None,
                 is_empty: bool = False):
        self._s_lin = s_lin
        self._s_ang = s_ang
        self._px = px
        self._py = py
        self._theta = theta
        self._vx = vx
        self._vy = vy
        self._omega = omega
        self._is_empty = is_empty

    @property
    def s_lin(self) -> float:
        return self._s_lin

    @property
    def s_ang(self) -> float:
        return self._s_ang

    @property
    def px(self) -> Optional[float]:
        return self._px

    @property
    def py(self) -> Optional[float]:
        return self._py

    @property
    def theta(self) -> Optional[float]:
        return self._theta

    @property
    def vx(self) -> Optional[float]:
        return self._vx

    @property
    def vy(self) -> Optional[float]:
        return self._vy

    @property
    def omega(self) -> Optional[float]:
        return self._omega

    @property
    def is_empty(self) -> bool:
        return self._is_empty

    @staticmethod
    def create_empty():
        return ActionPoint(
            s_lin=None,
            s_ang=None,
            px=None,
            py=None,
            theta=None,
            vx=None,
            vy=None,
            omega=None,
            is_empty=True
        )
