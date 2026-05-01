import numpy as np

class RobotState:
    def __init__(self) -> None:
        self.__q:   np.ndarray = None   # joints angle [rad]
        self.__dq:  np.ndarray = None   # joints velocity [rad/s]
        self.__ddq: np.ndarray = None   # joints acceleration [rad/s^2]
        self.__i:   np.ndarray = None   # joints current [A]
        self.__tau: np.ndarray = None   # joints torque [N m]
        self.__f:   np.ndarray = None   # TCP force [N]

    def __check_none(self, value):
        if value == None:
            raise ValueError('Incorrect value is None')

    def __check_nparray(self, value):
        if not type(value).__module__ == np.__name__:
            raise ValueError('Incorrect value is not np.array')

    @property
    def q(self) -> np.ndarray:
        return self.__q

    @q.setter
    def q(self, value):
        self.__check_nparray(value)
        self.__q = value

    @property
    def dq(self) -> np.ndarray:
        return self.__dq

    @dq.setter
    def dq(self, value: np.ndarray):
        self.__check_nparray(value)
        self.__dq = value

    @property
    def ddq(self) -> np.ndarray:
        return self.__ddq

    @ddq.setter
    def ddq(self, value: np.ndarray):
        self.__check_nparray(value)
        self.__ddq = value

    @property
    def i(self) -> np.ndarray:
        return self.__i

    @i.setter
    def i(self, value: np.ndarray):
        self.__check_nparray(value)
        self.__i = value

    @property
    def f(self) -> np.ndarray:
        return self.__f

    @f.setter
    def f(self, value: np.ndarray):
        self.__check_nparray(value)
        self.__f = value

    @property
    def tau(self) -> np.ndarray:
        return self.__tau

    @tau.setter
    def tau(self, value: np.ndarray):
        self.__check_nparray(value)
        self.__tau = value