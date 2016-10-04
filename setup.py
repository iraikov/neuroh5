from setuptools import setup

setup(
    name='neurograph',
    version='0.0.1',
    py_modules=['neurograph','lsnimport'],
    include_package_data=True,
    install_requires=[
        'click', 'h5py', 
    ],
    entry_points='''
        [console_scripts]
        neurograph=neurograph:cli
        importpairs=importpairs:cli
    ''',
)
