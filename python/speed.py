import numpy as np
import jax
import stravaio

def power(speed, grade, mass, cda, cxx, dt_efficiency):
    return 1/dt_efficiency * v * (cda*v*v + cxx + mass * 9.81 * np.sin(np.atan(grade)))

class RiderModel:
    def __init__(self, mass, Cda, Cxx, dt_efficiency):
        self.mass = mass
        self.Cda = 
