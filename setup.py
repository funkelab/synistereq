from setuptools import setup

setup(
    name='synistereq',
    version='0.1',
    description='Neurotransmitter classification requests',
    url='https://github.com/funkelab/synistereq',
    author='Funkelab',
    packages=[
        'synistereq',
        'synistereq.checkpoints',
        'synistereq.datasets',
        'synistereq.interfaces',
        'synistereq.loader',
        'synistereq.models',
        'synistereq.report',
        'synistereq.report.latex',
        'synistereq.repositories',
        ],
    scripts=[
        'synistereq/synister_request'
    ],
    package_data={"": ["*.ini","*.tex"]},
    include_package_data=True,
    install_requires=[])
