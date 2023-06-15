from setuptools import setup

package_name = 'map_cover'


setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/resource', ['resource/v7-LandCover-retrained-twice.h5']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Cognitive Robotics Lab, Technion',
    maintainer_email='karpase@gmail.com',
    description='Package for predicting cover of a map',
    license='FML',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'predict_node = map_cover.predict_node:main',
            'test_predict_node = map_cover.test_predict_node:main',
        ],
    },
)
