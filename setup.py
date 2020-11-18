from setuptools import setup
import time, os

packages = ['numpy==1.16.2',        
            'tensorflow==1.14.0',
            'gym',
            'opencv-python',
            'stable_baselines==2.10.0',
            'pyyaml',
            'pyqtgraph']
#            'PyQt5',
setup(
    name='rl',
    description='RL training library.',
    long_description='GUI for OpenAI Gym, scripts, configurations',
    version='0.2',
    packages=['rl'],
    scripts=[],
    author='Andrius Bernatavicius',
    author_email='andrius.bernatavicius@vanderlande.com',
    url='none',
    download_url='none',
    install_requires=packages
)

print ("Installation complete.\n")
