from setuptools import setup

setup(
    name='synistereq',
    version='0.1',
    description='Neurotransmitter classification requests',
    url='https://github.com/funkelab/synreq',
    author='Funkelab',
    packages=[
        'synistereq',
        'synistereq.interfaces',
        'synistereq.datasets',
        'synistereq.models',
        'synistereq.checkpoints',
        'synistereq.loader'
        ])
