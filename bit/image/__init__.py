# -*- coding: utf-8 -*-
"""
Collection of image processing functions, focused on astronomical images.
"""

def __autoEnableModules():
    import os
    import glob
    _dirname = os.path.dirname
    _join = os.path.join
    files = glob.glob(_join(_dirname(__file__),"*.py"))
    modules = [ os.path.basename(f)[:-3] for f in files ]
    return [ m for m in modules if not m[0] is "_" ]

__all__ = __autoEnableModules()

#import numpy as np
#
#class Image():
#    """
#    Define imagens por numpy arrays.
#    O foco de aplicacao da imagem aqui e de telescopio, se eh que isso interessa..
#    """
#    def __init__(self,data):
#        """
#        Initialize the class for 'data' image processing. 
#
#        """
#        super(Image,self).__init__( )
#
#        if isinstance(data,np.ndarray):
#            self._data = data.astype(np.float)
#        else:
##            oops.write("Oops: Dont know what your data is. Using \"NULL\" for data")
#            self._data = np.ndarray((),dtype=np.float)
#
##        self._undo = self._data.copy()
#        self._original = self._data.copy()
#        self._header = {}
#    
#        # Image data basic features
#        basicFeatures()
#
#
#class BasicFeatures(self):
#            """Extracts basic features from image data"""
#            def histogram():
#                pass
#            def min():
#                pass
#            def max():
#                pass
#            def quartile():
#                pass
#            def variance():
#                pass
#            def mode():
#                pass
#                
#class Header(dict):
#    """
#    Define Header por dicionarios
#    """
#    def __init__(self):
#        super(Header,self).__init__()
#        
#    
#class telescope():
#    """
#    Class with methos to handle astronomical images
#
#    It is important to note the very basic methods:
#    data()      : retrieve a copy of 'data'
#    header()    : retrieve a copy of 'header'
#    copy()      : make a copy of this object
#    setData()   : set 'data'   (ndarray)
#    setHeader() : set 'header' (dict)
#    
#    Also, a copy of (last) set 'data' is backed up, and the method 'refreshData()' 
#    does the refreshing to its original state.
#        
#    """
#    import pyfits
#    from io import fits
#    from matplotlib import pyplot as plt
#    
#    oops = open('ImageTelescopeOut.log','w')
#    
#    def __init__(self,data=None,header={}):
#        """
#        Initialize the class for 'data' image processing. 
#
#        """
#        if isinstance(data,np.ndarray):
#            __data__ = data.copy()
#        else:
#            oops.write("Oops: Dont know what your data is. Using \"NULL\" for data")
#            __data__ = np.ndarray((),dtype=np.float)
#
##        __undo__ = __data__.copy()
#        __original__ = __data__.copy()
#        __header__ = {}
#
#        
#    def setData(self,ndarray):
#        """
#        
#        """
#        self.__data__ = ndarray.astype(np.float)
#        self.__original__ = self.__data__.copy()
#        
#    def setHeader(self,header):
#        """
#        """
#        try:
#            self.__header__ = header.copy()
#        except:
#            try:
#                self.__header__ = header[:]
#            except:
#                self.__header__ = header
#            
#    def refreshData(self):
#        """
#        """
#        self.setData(self.__original__)
#        
#    def data(self):
#        """
#        """
#        return self.__data__
#        
#    def header(self):
#        """
#        """
#        return self.__header__
#        
#    def copy(self):
#        """
#        """
#        return self
#    
#    def fileRead(self,filename,hdu=0):
#        """
#        Open a FITS image file
#        
#        Input:
#         filename <str> : FITS image filename
#        """
#        file = pyfits.open(filename,memmap=True)
#        self.setData(file[hdu].data)
#        self.setHeader(file[hdu].header)
##        self.__data__ = file[hdu].data
##        self.__header__ = file[hdu].header
#        
#    def fileWrite(self,filename):
#        """
#        Write data to a FITS file
#        
#        Input:
#         filename <str> : FITS filename
#        """
#        pyfits.write(filename, self.__data__, self.__header__)
#        
#    def show(self):
#        """
#        Presents image
#        """
#        plt.imshow(self.__data__)
#        plt.show()
#
#        
#    ### TRATAMENTO/PRE-PROCESSING
#    # filtering
#    import smooth as s__
#    def filterMean(self,window=(3,3)):
#        """
#        Average/step filtering
#        """
#        self.__data__ = s__.average(self.__data__,window)
#        
#    def filterMedian(self,window=(3,3)):
#        """
#        Median filtering
#        """
#        self.__data__ = s__.median_filter(self.__data__,window)
#        
#    def filterGaussian(self,sigma=(2,2)):
#        """
#        Gaussian filtering
#        """
#        self.__data__ = s__.gaussian(self.__data__,sigma)
#    
#    def filterDirectional(self,size=5):
#        """
#        Directional smoothing
#        """
#        self.__data__ = s__.directional(self.__data__,size)
#        
#    # enhance
#    import rescaling as r__
#    def brightEqualization(): pass;
#    def brightCrop(): pass;
#    def brightSteping(): pass;
#    
#    # effects
#    import noise as n__
#    def addNoiseGaussian(): pass;
#    def addNoisePoisson(): pass;
#    def addNoiseSaltnPepper(): pass;
#    def addImage(): pass;
#    def addStamp(): pass;
#    
#    # features/props/signatures
#    import thresholding
#    def queryThreshold(): pass;
#    
