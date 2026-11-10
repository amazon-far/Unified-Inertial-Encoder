"""Lazy element-wise mapping over a sequence."""
import typing as tp

_T_In  = tp.TypeVar("_T_In")
_T_Out = tp.TypeVar("_T_Out")


class LazyMap(tp.Sequence[_T_Out], tp.Generic[_T_In, _T_Out]):
    def __init__(self, f: tp.Callable[[_T_In], _T_Out], xs: tp.Sequence[_T_In]):
        self.f = f
        self.xs = xs

    @tp.overload
    def __getitem__(self, i: int) -> _T_Out: ...
    @tp.overload
    def __getitem__(self, i: slice) -> tp.Sequence[_T_Out]: ...
        
    def __getitem__(self, i: int | slice) -> _T_Out | tp.Sequence[_T_Out]:
        if isinstance(i, slice):
            return [self.f(x) for x in self.xs[i]]
        return self.f(self.xs[i])

    def __len__(self):
        return len(self.xs)
