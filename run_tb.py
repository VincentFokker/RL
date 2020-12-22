import os, sys, subprocess, webbrowser, time

def kill_tensorboard():
    """Kills current running TB"""
    print('Closing current session of tensorboard.')
    if sys.platform == 'win32':
        os.system("taskkill /f /im  tensorboard.exe")


def launch_tensorboard(env_path):
    """Launches a new TB with given path"""
    # kill existing sessions
    kill_tensorboard()

    # Open the dir of the current env
    cmd = 'tensorboard.exe --logdir ' + env_path
    print('Launching tensorboard at {}'.format(env_path))
    DEVNULL = open(os.devnull, 'wb')
    subprocess.Popen(cmd, shell=True, stdout=DEVNULL, stderr=DEVNULL)
    time.sleep(5)
    webbrowser.open_new_tab(url='http://localhost:6006/#scalars&_smoothingWeight=0.960')



_env_path = 'rl/trained_models/NewConveyor10/20201029_1530'
launch_tensorboard(_env_path)