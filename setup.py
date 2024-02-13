"""setup file for the project."""
# code gratefully take from https://github.com/navdeep-G/setup.py

# Note: To use the 'upload' functionality of this file, you must:
#   $ pipenv install twine --dev

import io
import os
import sys
from shutil import rmtree
import versioneer

from setuptools import find_packages, setup, Command

# Package meta-data.
NAME = 'xiRT'
DESCRIPTION = \
    'xiRT: Multi-dimensional Retention Time Prediction for Linear and Crosslinked Peptides.'
URL = 'https://rappsilber-laboratory.github.io/xiRT/'
URL_DOKU = "https://xirt.readthedocs.io/en/latest/"
URL_GITHUB = "https://github.com/Rappsilber-Laboratory/xiRT"
URL_ISSUES = "https://github.com/Rappsilber-Laboratory/xiRT/issues"
EMAIL = 'sven.giese@hpi.de'
AUTHOR = 'Sven Giese'
REQUIRES_PYTHON = '>=3.6.0'
KEYWORDS = ["xiRT", "Proteomics", "Crosslinking", "machine learning", "Retention Time Prediction",
            "Chromatography", "Peptides"]
RAPPSILBER_SOFTWARE = "https://www.rappsilberlab.org/software/"
# What packages are required for this module to be executed?
REQUIRED = [
    'numpy==1.24.3',
    'pandas==2.0.3',
    'tensorflow==2.13.0',
    'seaborn==0.12.2',
    'xlwt==1.3.0',
    'graphviz==0.20.1',
    'pydot==1.4.2',
    'pyyaml==6.0.1',
    'pyteomics==4.6',
    'scikit-learn==1.2.2',
    'tqdm==4.66.0',
    'biopython==1.81',
    'pydot==1.4.2',
    'palettable==3.3.3',
    'statannot==0.2.3'
]

# What packages are optional?
# 'fancy feature': ['django'],}
EXTRAS = {
    'develop': [
        'pytest~=2.8.6',
        'flake8~=2.5.2'
    ]
}

# The rest you shouldn't have to touch too much :)
# ------------------------------------------------
# Except, perhaps the License and Trove Classifiers!
# If you do change the License, remember to change the Trove Classifier for that!

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's __version__.py module as a dictionary.
about = {}
project_slug = "xirt"
# with open(os.path.join(here, project_slug, '__version__.py')) as f:
#     exec(f.read(), about)


class UploadCommand(Command):
    """Support setup.py upload."""

    description = 'Build and publish the package.'
    user_options = []

    @staticmethod
    def status(s):
        """Print things in bold."""
        print('\033[1m{0}\033[0m'.format(s))

    def initialize_options(self):
        """Init options."""
        pass

    def finalize_options(self):
        """Finalize method."""
        pass

    def run(self):
        """Run method."""
        try:
            self.status('Removing previous builds…')
            rmtree(os.path.join(here, 'dist'))
        except OSError:
            pass

        self.status('Building Source and Wheel (universal) distribution…')
        os.system('{0} setup.py sdist bdist_wheel --universal'.format(sys.executable))

        self.status('Uploading the package to PyPI via Twine…')
        os.system('twine upload dist/*')

        self.status('Pushing git tags…')
        os.system('git tag v{0}'.format(about['__version__']))
        os.system('git push --tags')

        sys.exit()


# Where the magic happens:
setup(
    name=NAME,
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    project_urls={
        "Bug Tracker": URL_ISSUES,
        "Source Code": URL_GITHUB,
        "Documentation": URL_DOKU,
        "Homepage": URL,
        "Related Software": RAPPSILBER_SOFTWARE},
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    # If your package is a single module, use this instead of 'packages':
    # py_modules=['mypackage'],

    # 'mycli=mymodule:cli'
    entry_points={'console_scripts': ["xirt=xirt.__main__:main"],
                  },
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license='Apache License 2.0',
    keywords=KEYWORDS,
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: Implementation :: PyPy'
    ],
)
