from .driver import camera, stream
from picar import back_wheels, front_wheels
import picar

picar.setup()
db_file = "/home/pi/SunFounder_PiCar-V/remote_control/remote_control/driver/config"
fw = front_wheels.Front_Wheels(debug=False, db=db_file)
bw = back_wheels.Back_Wheels(debug=False, db=db_file)
cam = camera.Camera(debug=False, db=db_file)
cam.ready()
bw.ready()
fw.ready()

fwOffset = 90
bwStatus = 0

dt = 250
commandHistory = []
maxHistory = 10

print(stream.start())

def update(tf1, tf2, ts, v, tb):
    front(ts)
    back(v)
    frontCam(tf1, tf2)
    backCam(tb)

    if(len(commandHistory) >= maxHistory):
        commandHistory = commandHistory[:maxHistory-1]
    commandHistory.append((tf1, tf2, ts, v, tb))
    
def front(ts):
    #Forward Steering
    fw.turn(fwOffset + ts)

def back(v):
    #Back wheel speed
    bw.speed = abs(v)

    if(v < 0):
        bw.backward()
        bwStatus = -1
    elif(v > 0):
        bw.forward()
        bwStatus = 1
    else:
        bw.stop()
        bwStatus = 0

def frontCam(tf1, tf2):
    #Front Camera
    if(tf1 < 0):
        cam.turn_up(abs(tf1))
    elif(tf1 > 0):
        cam.turn_down(abs(tf1))
    else:
        pass

    #Back Camera
    if(tf2 < 0):
        cam.turn_left(abs(tf2))
    elif(tf2 > 0):
        cam.turn_right(abs(tf2))
    else:
        pass


def backCam(tb):
    pass