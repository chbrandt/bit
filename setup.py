#!/usr/bin/env python
# ===============================================
# Setup (build/install) script
#
# Author: Carlos Brandt
#
# Last Change: Jun.2012
# ===============================================
import sys
import os

#from distribute_setup import use_setuptools
#use_setuptools()
from setuptools import setup
import __version__

NAME                = "bit"
VERSION             = __version__.VERSION
KEYWORDS            = "image catalogs fits segmentation svm"
DESC                = "Python image processing and visualization tools"
AUTHOR              = "Carlos Brandt"
AUTHOR_EMAIL        = "carloshenriquebrandt@gmail.com"
URL                 = "https://github.com/chbrandt/bit"

install_deps = ['pyfits>=2.3']

joinpath = os.path.join

setup(  name = NAME,
        version = VERSION,
        maintainer = AUTHOR,
        maintainer_email = AUTHOR_EMAIL,
        url = URL,
#        install_requires = install_deps,
        package_dir = {'bit':'.'},
        packages = ['bit',
                    'bit.classify',
                    'bit.graphics',
                    'bit.image',
                    'bit.io',
                    'bit.pipelines',
                    'bit.tables',
                    ],
        package_data = {'bit' : ['data/*.jpg',
                                'data/*.png',
                                'data/*.fits',
                                ]
                        }
)


#-------------------------------
#def check_dependencies(deps={}):
#    check_flag = True
#    for mod,msg in deps.items():
#        try:
#            __import__(mod)
#        except ImportError:
#            print "Module/Package %s not found. Check installation [%s]." % (mod,msg)
#            check_flag = False
#    
#    return check_flag
#
#deps = {'dicom':'(v0.9.5) http://code.google.com/p/pydicom/',
#        'PIL':'(v1.1.6) http://www.pythonware.com/products/pil/',
#        'vtk':'(v5.2.1) http://www.vtk.org/',
#        'numpy':'(v1.3.0) http://numpy.scipy.org/',
#        'scipy':'(v0.7.0) http://scipy.org/',
#        'PyQt4':'(v4.6) http://www.riverbankcomputing.co.uk/',
#        'PyQt4.QSci':'Qscintilla2 (v2.4) http://www.riverbankcomputing.co.uk/',
#        'matplotlib':'(v0.99.0) http://matplotlib.sourceforge.net/',
#        }
#
#if not check_dependencies(deps):
#    print "Dependencies should be first satisfied to conclude installation."
#    sys.exit(1)

