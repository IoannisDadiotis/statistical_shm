# statistical_shm
Dynamic identification and structural health monitoring of real structures using statistical time series analysis. Project as part of the taught [Dynamic identification and SHM] at the University of Patras.

## The systems

## Dynamic identification
The developed force sensory system (FSS) is based on strain gauge-sensing and targeted for the fabric gripper that is presented in:

* PN Koustoumpardis, S Smyrnis, NA Aspragathos, [A 3-Finger Robotic Gripper for Grasping Fabrics Based on Cams-Followers Mechanism], International Conference on Robotics in Alpe-Adria Danube Region, 2017

The measurement chain of the FSS consists of:

* 4 strain gauges 350 Ohm Tokyo Instruments FLAB-3-350-11-1LJB-F
* 2 custom half Wheatstone bridges (custom made with 350 Ohm resistors)
* A Load Cell/Wheatstone Shield Amplifier bought by [Robotshop.com]. The shield is equipped with an AD8426 amplifier from Analog Devices that achieves gain up to 1000.
* A Nucleo F334R8 microcontroller.

<!--p float="left">
  <img src="https://user-images.githubusercontent.com/75118133/159372939-beaf94a2-fa9c-4b10-b6cc-da1e09dafda9.png" width="250" />
  <img width="10" />
  <img src="https://user-images.githubusercontent.com/75118133/159373044-4143eb60-5efa-44cb-ad3a-6fe17e54c543.png" width="300" /> 
</p-->

The sensing principle of the FSS is shown below:


<img src="https://user-images.githubusercontent.com/75118133/159374038-3470c8cd-0274-4bee-ba54-6d72d12e9dba.png" width="450" />

<p float="left">
  <img src="https://user-images.githubusercontent.com/75118133/159375447-395f4a6d-1de8-4425-b5af-53eb7837ac1e.png" width="350" />
  <img width="50" />
  <img src="https://user-images.githubusercontent.com/75118133/159375021-d0e04246-2cdb-4cbb-838e-31c801a3b4b3.png" width="200" /> 
</p>

All components are installed onboard a small mobile robot which is equipped with a esp8266 Wi-Fi module for wireless communication with a laptop.

<!--![image](https://user-images.githubusercontent.com/75118133/159376847-4b2ff712-d9be-4eac-9caa-80cc84c042f4.png)-->

https://user-images.githubusercontent.com/75118133/159466400-35142a2c-c064-4421-b7d7-cb027e79a0fe.mp4

## Structural health monitoring
The FSS presente above was used for cooperative fabric transportation by two wheeled mobile robot in a leader-follower formation. Only the follower is equipped with the FSS and used the provided fabric force feedback to follow the leader.

<img src="https://user-images.githubusercontent.com/75118133/159376304-20202a23-c892-4d47-8cb0-8b0c0be8201a.png" width="450" />

The force control system of the follower is comprised of two parallel PID controllers.

<p float="left">
  <img src="https://user-images.githubusercontent.com/75118133/159375837-40476073-5735-401d-93b3-7df05c143125.png" width="300" />
  <img width="10" />
  <img src="https://user-images.githubusercontent.com/75118133/159376542-c8e2db9f-2235-49e5-8044-274fb0c01a08.png" width="250" /> 
</p>

https://user-images.githubusercontent.com/75118133/159466671-b49ed355-f67b-4f78-b4b5-3cac6e75bfd1.mp4

## Running the code

* The onboard Nucleo MCU accepts code in C++ and can be programmed through the [mbed online compiler] while in cabled connection with the laptop. The MCU keeps the code that was last passed and runs it every time you switch it on or press the reset button.
* The Wifi communication between laptop and robot as well as the force control scheme is implemented in Python.
* To read the data from the FSS with the robot not moving and connected through usb cable with the laptop, you have to pass the code of `src/mbed/Static_experiments_Nucleo.txt` to the MCU throught the online compiler.
* For the wireless force control experiments (connection through Wi-Fi) the procedure is:
  - Pass the code at `src/mbed/Control experiments_Nucleo_follower.txt` and `src/mbed/Control experiments_Nucleo_leader.txt` to the follower and leader robots, respectively. Remember the follower is equipped with the FSS.
  - Run the code `src/wifi_communication/server.py` on the laptop
  - Switch-on or reset the robots.
  - To succesfully connect the robots with the laptop their IPs should be properly configured in the above files.
            
## Results
This work is related with the above publications:

* ID Dadiotis, JS Sakellariou, PN Koustoumpardis, [Development of a low-cost force sensory system for force control via small grippers of cooperative mobile robots used for fabric manipulation]. Adv. Mechanisms and Machine Science 2021, 102, 47–58.
* I. Dadiotis Development of a force sensory system for the control and coordination of mobile robots
handling fabrics, Nemertes: Institutional Repository of University of Patras, Patras, Greece, 2020 (in Greek)
http://hdl.handle.net/10889/13747

## Contributors and acknowledgments
This project was conducted at the Department of Mechanical Engineering and Aeronautics at the University of Patras, Greece in 2020 by Ioannis Dadiotis as part of the [Dynamic identification and SHM] course by Prof. S. Fassois and Ass. professor J. Sakellariou of the [SMSA lab].

![image](https://user-images.githubusercontent.com/75118133/159381029-ff271c1e-f995-42a1-a11a-2c50890c7e5e.png)

[SMSA lab]: https://sites.google.com/g.upatras.gr/smsa-lab/home
[Dynamic identification and SHM]: https://www.mead.upatras.gr/en/courses/domiki-akeraiotita-kataskeuwn/

