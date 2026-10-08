from time import time

import mujoco
import mujoco.viewer

def key_callback(keycode):
    global keytime
    if keycode == 265:
        data.ctrl[0]=2
        data.ctrl[1]=2
    elif keycode == 264:
        data.ctrl[0]=-2
        data.ctrl[1]=-2
    elif keycode == 263:
        data.ctrl[0]=-0.5
        data.ctrl[1]=0.5
    elif keycode == 262:
        data.ctrl[0]=0.5
        data.ctrl[1]=-0.5
    keytime=time()

FPS=30
timestep=1/FPS
keytime=time()

model = mujoco.MjModel.from_xml_path("smartcar/scene1.xml")
data = mujoco.MjData(model)
viewer = mujoco.viewer.launch_passive(model,data,show_left_ui=False,show_right_ui=False,key_callback=key_callback)
viewer.cam.distance = 12.0
renderer = mujoco.Renderer(model, height=240, width=320)

def run(runFunc,manual=False):
    timestamp=time()
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        if time()-timestamp<timestep:
            continue
        timestamp=time()
        renderer.update_scene(data,camera="carCamera")
        img = renderer.render()
        imu=data.sensordata
        leftspeed,rightspeed=runFunc(img,imu)
        if manual:
            if time()-keytime>0.1:
                data.ctrl[0]=0
                data.ctrl[1]=0
            continue
        data.ctrl[0]=leftspeed
        data.ctrl[1]=rightspeed
    renderer.close()
    viewer.close()

