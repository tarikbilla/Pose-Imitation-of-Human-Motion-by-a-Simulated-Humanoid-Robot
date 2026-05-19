from controller import Robot

robot = Robot()

timestep = int(robot.getBasicTimeStep())

# Get motors
headYaw = robot.getDevice("HeadYaw")
headPitch = robot.getDevice("HeadPitch")

# Move head
headYaw.setPosition(1.0)
headPitch.setPosition(0.3)

while robot.step(timestep) != -1:
    pass
