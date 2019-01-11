import tinyik
import  numpy as np

arm = tinyik.Actuator(['z', [1.,0.,0.], 'z', [1.,0.,0.]])

print(arm.angles)
print(arm.ee)

arm.angles = [np.pi/6, np.pi/3]
print(arm.ee)

arm.ee = [2/np.sqrt(2), 2/np.sqrt(2), 0.]
print(arm.angles)

print(np.round(np.rad2deg(arm.angles)))