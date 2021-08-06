from setuptools import setup

setup(
    name='synistereq',
    version='0.1',
    description='Neurotransmitter classification requests',
    url='https://github.com/funkelab/synistereq',
    author='Funkelab',
    packages=[
        'synistereq',
        'synistereq.interfaces',
        'synistereq.datasets',
        'synistereq.models',
        'synistereq.checkpoints',
        'synistereq.loader',
        'synistereq.report',
        'synistereq.report.latex'
        ],
    scripts=[
        'synistereq/synister_request'
    ],
    package_data={"": ["*.ini","*.tex"]},
    include_package_data=True,
    install_requires=[])
