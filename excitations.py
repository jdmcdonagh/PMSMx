import math
import numpy as np
from scipy.spatial.transform import Rotation

class SupplyCurrentDensity:
    def __init__(self):
        self.cw = 1.
        self.amp = 1.
        self.omega = 1.
        self.t = 0.
        
        self.yl = 3.2223088765338/1000.
        self.zl = 40./1000.
        self.m = 0.577

    def eval(self, coord):
        """PMSM Current Density Excitation"""
        values = np.zeros((3, coord.shape[1]))
        
        yl, zl, m = self.yl, self.zl, self.m
        cw, amp, omega, t = self.cw, self.amp, self.omega, self.t
        
        phase_a = amp*math.sin( omega*t )
        phase_b = amp*math.sin( omega*t + (2*math.pi/3) )
        phase_c = amp*math.sin( omega*t - (2*math.pi/3) )
        
        #print(amp, omega, t, phase_a, phase_b, phase_c)
        
        # apply load for aligned domain
        for i in range(coord.shape[1]):
            x, y, z = coord[0][i], coord[1][i], coord[2][i] 
            
            # which sector coordinate is in, get rotation angle
            if y < m*x and y > -1*m*x: #S0
                rotangle = 0
                pmag = phase_a
                coilcw = -1
            elif x > 0 and y > m*x: #S1
                rotangle = 60
                pmag = phase_b
                coilcw = -1
            elif y > -1*m*x and x < 0: #S2
                rotangle = 120
                pmag = phase_c
                coilcw = 1
            elif y > m*x and y < -1*m*x: #S3
                rotangle = 180
                pmag = phase_a
                coilcw = 1
            elif x < 0 and y < m*x: #S4
                rotangle = 240
                pmag = phase_b
                coilcw = 1
            elif y < -1*m*x and x > 0: #S5
                rotangle = 300
                pmag = phase_c
                coilcw = -1
            
            # rotate coordinates to S0
            r = Rotation.from_euler('z', -1*rotangle, degrees=True)
            vec = r.apply([x, y, z])
            x, y, z = vec[0], vec[1], vec[2]
            
            # calculate which region in S0 the point is in
            # R1 (linear)
            if (y <= yl and y >= -1*yl) and (z > zl):
                values[1, i] = cw*coilcw*-1*pmag
                 
            # R2 (rotation)
            elif y < -1*yl and z > zl:
                dy, dz = y + yl, z - zl
                r = math.sqrt(dy**2 + dz**2)
                values[1, i] = cw*coilcw*-1*pmag*(dz/r)
                values[2, i] = cw*coilcw*pmag*(dy/r)
                
            # R3 (linear)
            elif (z <= zl and z >= -1*zl) and (y < -1*yl):
                values[2, i] = cw*coilcw*-1*pmag
                
            # R4 (rotation)
            elif y < -1*yl and z < -1*zl:
                dy, dz = y + yl, z + zl
                r = math.sqrt(dy**2 + dz**2)
                values[1, i] = cw*coilcw*-1*pmag*(dz/r)
                values[2, i] = cw*coilcw*pmag*(dy/r)
                
            # R5 (linear)
            elif (y <= yl and y >= -1*yl) and (z < -1*zl):
                values[1, i] = cw*coilcw*pmag
                
            # R6 (rotation)
            elif y > yl and z < -1*zl:
                dy, dz = y - yl, z + zl
                r = math.sqrt(dy**2 + dz**2)
                values[1, i] = cw*coilcw*-1*pmag*(dz/r)
                values[2, i] = cw*coilcw*pmag*(dy/r)
                
            # R7 (linear)
            elif (z <= zl and z >= -1*zl) and (y > yl):
                values[2, i] = cw*coilcw*pmag
                
            # R8 (rotation)
            elif y > yl and z > zl:
                dy, dz = y - yl, z - zl
                r = math.sqrt(dy**2 + dz**2)
                values[1, i] = cw*coilcw*-1*pmag*(dz/r)
                values[2, i] = cw*coilcw*pmag*(dy/r)
            
            # rotate the excitation back to original sector
            rot = Rotation.from_euler('z', rotangle, degrees=True)
            
            vec = [values[0, i], values[1, i], values[2, i]]
            values[0, i] = rot.apply(vec)[0]
            values[1, i] = rot.apply(vec)[1]
            values[2, i] = rot.apply(vec)[2]
            
        return values
    
    
class PMMagnetization:
    def __init__(self):
        self.sign = 1.
        self.mag = 1.
        self.m1, self.m2 = 0.727, 3.08
        
    def eval(self, coord):
        """PMSM Magnetization Excitation"""
        values = np.zeros((3, coord.shape[1]))
        
        m1, m2 = self.m1, self.m2
        mag, sign = self.mag, self.sign
        
        # apply load for aligned domain
        for i in range(coord.shape[1]):
            x, y, z = coord[0][i], coord[1][i], coord[2][i] 
            
            # which sector coordinate is in, get rotation angle
            # S0
            if y > 0 and y <= m1*x:
                rotangle = (360/10*0)+18
                inout = 1
            # S1
            elif y > m1*x and y <= m2*x:
                rotangle = (360/10*1)+18
                inout = -1
            # S2
            elif y > m2*x and y >= -1*m2*x:
                rotangle = (360/10*2)+18
                inout = 1
            # S3
            elif y < -1*m2*x and y >= -1*m1*x:
                rotangle = (360/10*3)+18
                inout = -1
            # S4
            elif y < -1*m1*x and y >= 0:
                rotangle = (360/10*4)+18
                inout = 1
            # S5
            elif y < 0 and y >= m1*x:
                rotangle = (360/10*5)+18
                inout = -1
            # S6
            elif y < m1*x and y >= m2*x:
                rotangle = (360/10*6)+18
                inout = 1
            # S7
            elif y < m2*x and y <= -1*m2*x:
                rotangle = (360/10*7)+18
                inout = -1
            # S8
            elif y > -1*m2*x and y <= -1*m1*x:
                rotangle = (360/10*8)+18
                inout = 1
            # S9
            elif y > -1*m1*x and y <= 0:
                rotangle = (360/10*9)+18
                inout = -1
            
            # create unit vector in correct direction
            r = Rotation.from_euler('z', rotangle, degrees=True)
            unitvec = r.apply([1, 0, 0])
            result = sign*mag*inout*unitvec
            
            values[0, i] = result[0]
            values[1, i] = result[1]
            values[2, i] = result[2]
            
        return values