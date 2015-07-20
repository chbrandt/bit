# -*- coding:utf-8 -*-
"""
.. module:: wavebands
    :synopsis: Multiwavelength data plot

.. moduleauthor:: Carlos Brandt <carloshenriquebrandt@gmail.com>
"""
#
# Module to create a plot for surveys spectral sensitivity, all wavebands:
# - Radio
# - Millimeter
# - Infrared
# - Optical
# - UV
# - EUV
# - X-ray
# - Gamma-ray

from log import *
import inspect

# Let's write down the structure that defines the (wave)bands.
# This is temporary, it should go to a config file in the near future.
# Each entry in WAVEBANDS (structure/config) is headed by the waveband name
#  (as the key) pointing to a dictionary with the following key/values:
#  - unit : (length,frequency/energy) unit in which 'min/max' are expressed
#  - min  : limiting minimum value
#  - max  : limiting maximum value
from collections import namedtuple
_WAVEBANDS = {
        'radio' : {
            'unit' : 'mm',
            'min' : 10,
            'max' : 1000 # support inf/nan
        },
        'millimeter' : {
            'unit' : 'mm',
            'min' : 0.1,
            'max' : 10
        },
        'infrared' : {
            'unit' : 'um',
            'min' : 1,
            'max' : 100
        },
        'optical' : {
            'unit' : 'angstrom',
            'min' : 3000,
            'max' : 10000
        },
        'uv' : {
            'unit' : 'nm',
            'min' : 100,
            'max' : 300
        },
        'euv' : {
            'unit' : 'nm',
            'min' : 10,
            'max' : 100
        },
        'xray' : {
            'unit' : 'angstrom',
            'min' : 0.1,
            'max' : 100
        },
        'gammaray' : {
            'unit' : 'angstrom',
            'min' : 0.001, # support inf/nan
            'max' : 0.1
        }
}

from astropy import units

class Waveband(object):
    """
    Waveband definition, limits and adequate units
    """
    _frequency_unitname = 'Hz'
    _energy_unitname = 'eV'

    def __init__(self,name,min,max,unit):
        super(Waveband,self).__init__()
        assert isinstance(name,str) and name is not ''
        self._name = name
        self._unit = units.Unit(unit,parse_strict='silent')
        _min = float(min)
        _max = float(max)
        assert _min < _max
        self._min = _min
        self._max = _max
        self._range = [float(min), float(max)] * self._unit

        self._unit_nu = units.Unit(self._frequency_unitname)
        self._unit_E = units.Unit(self._energy_unitname)

    def __str__(self):
        n = self.name
        mn = float(self.min.value)
        mx = float(self.max.value)
        u = str(self.unit)
        return '%15s %.3e %.3e %s' % (n,mn,mx,u)

    @property
    def name(self):
        """
        Waveband name
        """
        return self._name

    @property
    def unit(self):
        """
        Return the default unit
        """
        return self._unit

    def min(self,unit=None):
        """
        Return the lower limiting value
        """
        _m = min(self.limits(unit))
        return _m

    def max(self,unit=None):
        """
        Return the upper limiting value
        """
        _m = max(self.limits(unit))
        return _m

    def limits(self,unit=None):
        """
        Return waveband limits as a astropy.units.Quantity pair
        """
        if unit is None:
            return self._range
        _lim =  self.convert(self._range,units.Unit(unit))
        _lmin = min(_lim)
        _lmax = max(_lim)
        _q = units.Quantity([_lmin,_lmax])
        return _q

    def limits_wavelength(self,unit=None):
        """
        Convenient function to retrieve limits in the default wavelength unit
        """
        _unit = self._unit if unit is None else units.Unit(unit)
        return self.limits(_unit)

    def limits_frequency(self,unit=None):
        """
        Convenient function to retrieve limits in the default frequency unit
        """
        _unit = self._unit_nu if unit is None else units.Unit(unit)
        return self.limits(_unit)

    def limits_energy(self,unit=None):
        """
        Convenient function to retrieve limits in the default energy unit
        """
        _unit = self._unit_E if unit is None else units.Unit(unit)
        return self.limits(_unit)

    @staticmethod
    def convert(quantities,unit):
        """
        Convenient function to convert between equivalent quantities
        """
        return (quantities).to(unit,equivalencies=units.spectral())

class Wavebands(list):
    def __init__(self):
        super(Wavebands,self).__init__()
        self._unit = None

    def append(self,band):
        assert isinstance(band,Waveband)
        super(Wavebands,self).append(band)

    @property
    def min(self):
        """
        Retrieves the minimum among all wavebands
        """
        _min = float('+inf')
        for _b in self:
            _min = min(_min,_b.min(self._unit))
        assert _min >= 0.0, "_min is %f" % _min
        return _min

    @property
    def max(self):
        _max = float('-inf')
        for _b in self:
            _max = max(_max,_b.max(self._unit))
        assert _max > 0.0, "_max is %f" % _max
        return _max

    def __getUnit__(self):
        return self._unit

    def __setUnit__(self,unit=None):
        if unit is not None:
            unit = units.Unit(unit)
        self._unit = unit

    unit = property(__getUnit__,__setUnit__,
            doc="Get/Set default unit")

def findLimit(iterable,lim,func,lvl=0):
    """
    Find the limit (min,max) value within an iterable structure.

    Args:
        iterable: iterable object (e.g, [])
            Iterable object where values, one by one, will be compared.
            Object can be a flat or nested association of iterable objects.
        lim: comparable value (e.g, float)
            Initial limiting value to compare. Assume it as a boundary value.
        func: function (e.g, min)
            Comparative function to apply for any pair of values.
            ``func`` should take two values and return the succesfull one.

    Returns:
        Value within ``iterable`` that succeds comparison by ``func``.

    Example:
        >>> a = range(11)
        >>> from random import shuffle
        >>> shuffle(a)
        >>> lmin = findLimit(a,11,min)
        >>> lmax = findLimit(a,-1,max)
        >>> lmin==0
        True
        >>> lmax==10
        True
    """
    _shift = lvl*'\t'

    frame = inspect.currentframe()
    args,_,_,values = inspect.getargvalues(frame)
    log.debug(_shift+'Input arguments:')
    for arg in args:
        log.debug(_shift+'\t%s : %s' % (arg,values[arg]))

    log.debug(_shift+'Iterate over: %s' % str(iterable))
    log.debug(_shift+'\titerable object %s' % id(iterable))
    for it in iterable:
        log.debug(_shift+'\tcurrent it: %s' % str(it))
        log.debug(_shift+'\tinput lim: %s' % str(lim))
        if isinstance(it,(list,tuple)):
            log.debug(_shift+'\tnested structure (%s)' % type(it))
            it = findLimit(it,lim,func,lvl=lvl+1)
            log.debug(_shift+'\treturned limit: %s' % str(it))
            log.debug(_shift+'\tlim object %s' % id(it))
        lim = func(it,lim)
        log.debug(_shift+'\tcurrent lim: %s' % lim)
        log.debug(_shift+'\tlim object %s' % id(lim))
    return lim

# The core of the plot function -- annotation and axis cloning -- were
#  taken from stackoverflow's post by Joe Kington:
#  http://stackoverflow.com/a/3919443/687896
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
def sed(data):
    """
    """
    bands = Wavebands()
    for band,lims in _WAVEBANDS.items():
        _wb = Waveband(band,lims['min'],lims['max'],lims['unit'])
        bands.append(_wb)

    main_unit = 'GHz'

    _x,_y,_u = zip(*data)
    print _x,_y
    xu = []
    for xi,ui in zip(_x,_u):
        _xx = xi * units.Unit(ui)
        _xx = _xx.to(main_unit,equivalencies=units.spectral())
        print _xx
        xu.append(_xx.value)
    yu = []
    for yi in _y:
        yu.append(yi)

    #-- Plot the results ------------------------------------------------------
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Give ourselves a bit more room at the bottom
    plt.subplots_adjust(bottom=0.2)
    plt.xscale('log')

    bands.unit = main_unit
    ax.set_xlim(bands.min.value,bands.max.value)
    print "",yu
    print 'MIN'
    _ymin = findLimit(yu,lim=float('+inf'),func=min)
    print _ymin
    _ymin *= 0.9
    print 'MAX'
    _ymax = findLimit(yu,lim=float('-inf'),func=max)
    print _ymax
    _ymax *= 1.1
    ax.set_ylim(_ymin,_ymax)
    print len(xu)
    print len(yu)
    assert len(xu)==len(yu)
    for i in xrange(len(xu)):
        _x = xu[i]
        _y = yu[i]
        try:
            _LIXO = iter(_x)
            assert len(_x)==len(_y)
            ax.plot(_x, _y, 'b-', lw=5)
        except:
            print _x
            print _y
            ax.plot(_x, _y, 'b*')
    ax.grid(True)

    # Drop the bottom spine by 40 pts
    ax.spines['bottom'].set_position(('outward', 40))

    # Make a second bottom spine in the position of the original bottom spine
    make_second_bottom_spine(label='Wavebands')

    # Annotate the groups
#    for band in bands:
#        annotate_group(band,main_unit)

    plt.xlabel('Frequency(%s)' % main_unit)
    plt.ylabel('Depth')
    plt.title('Spectral depth')

    #plt.show()
    return plt

# --------------------------------------------------------------------------
def annotate_group(band, unit, ax=None):
    """Annotates a span of the x-axis"""
    def annotate(ax, name, left, right, y, pad):
        arrow = ax.annotate(name,
                xy=(left, y), xycoords='data',
                xytext=(right, y-pad), textcoords='data',
                annotation_clip=False, verticalalignment='top',
                horizontalalignment='center', linespacing=2.0,
                arrowprops=dict(arrowstyle='-', shrinkA=0, shrinkB=0,
                        connectionstyle='angle,angleB=90,angleA=0,rad=5')
                )
        return arrow
    if ax is None:
        ax = plt.gca()
    ymin = ax.get_ylim()[0]
    ypad = 0.01 * np.ptp(ax.get_ylim())

    name = band.name
    xspan = band.limits(unit=unit).value
    xcenter = np.mean(xspan)
    print name,xspan,xcenter
    left_arrow = annotate(ax, name, xspan[0], xcenter, ymin, ypad)
    right_arrow = annotate(ax, name, xspan[1], xcenter, ymin, ypad)

    return left_arrow, right_arrow
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
def make_second_bottom_spine(ax=None, label=None, offset=0, labeloffset=20):
    """Makes a second bottom spine"""
    if ax is None:
        ax = plt.gca()
    second_bottom = mpl.spines.Spine(ax, 'bottom', ax.spines['bottom']._path)
    second_bottom.set_position(('outward', offset))
    ax.spines['second_bottom'] = second_bottom

    if label is not None:
        # Make a new xlabel
        ax.annotate(label,
                xy=(0.5, 0), xycoords='axes fraction',
                xytext=(0, -labeloffset), textcoords='offset points',
                verticalalignment='top', horizontalalignment='center')
# --------------------------------------------------------------------------


# def create_plotgrid(shape=(1,7)):
#     """
#     Create a grid of (sub)plots, each for a each waveband
#     """
#
#     import numpy as np
#     from matplotlib import pyplot as plt
#     fig,axs = plt.subplots(1,8, sharex=False, sharey=True)
# #    fig.set(figheight=3,figwidth=9)
#
#     y_quantity = 'Depth'
#     figure_title = '%s - Waveband relation' % y_quantity
#
#     fig.suptitle(figure_title)
#     fig.subplots_adjust(wspace=0,top=0.8,bottom=0.2)
#
#     ax_atLeft = axs[0]
#     ax_atLeft.set(ylabel=y_quantity)
#     ax_atLeft.set(ylim=(0,1))
#     ax_atLeft.grid(True)
#
#     for ax in axs:
#         for label in ax.get_yticklabels():
#             label.set_visible(False)
#         for label in ax.get_xticklabels():
#             label.set_fontsize(10)
#             label.set_rotation('vertical')
#
#     axs[0].set(title='Radio')
#     radio_xlim=(1E9,1E7)
#     axs[0].set(xlim=radio_xlim)
#     radio_xticks=np.linspace(*radio_xlim,num=5)[:-1]
#     axs[0].set_xticks(radio_xticks)
#
#     axs[1].set(title='Millimeter')
#     mm_xlim=(1E7,1E5)
#     axs[1].set(xlim=mm_xlim)
#     mm_xticks=np.linspace(*mm_xlim,num=5)[:-1]
#     axs[1].set_xticks(mm_xticks)
#
#     axs[2].set(title='Infrared')
#     ir_xlim=(1E5,1E3)
#     axs[2].set(xlim=ir_xlim)
#     ir_xticks=np.linspace(*ir_xlim,num=5)[:-1]
#     axs[2].set_xticks(ir_xticks)
#
#     axs[3].set(title='Optical')
#     opt_xlim=(1E3,3E2)
#     axs[3].set(xlim=opt_xlim)
#     opt_xticks=np.linspace(*opt_xlim,num=5)[:-1]
#     axs[3].set_xticks(opt_xticks)
#
#     axs[4].set(title='UV')
#     uv_xlim=(3E2,1E2)
#     axs[4].set(xlim=uv_xlim)
#     uv_xticks=np.linspace(*uv_xlim,num=5)[:-1]
#     axs[4].set_xticks(uv_xticks)
#
#     axs[5].set(title='EUV')
#     euv_xlim=(1E2,10)
#     axs[5].set(xlim=euv_xlim)
#     euv_xticks=np.linspace(*euv_xlim,num=5)[:-1]
#     axs[5].set_xticks(euv_xticks)
#
#     axs[6].set(title='X-ray')
#     xr_xlim=(10,1E-2)
#     axs[6].set(xlim=xr_xlim)
#     xr_xticks=np.linspace(*xr_xlim,num=5)[:-1]
#     axs[6].set_xticks(xr_xticks)
#
#     axs[7].set(title='Gamma-ray')
#     gr_xlim=(1E-2,1E-5)
#     axs[7].set(xlim=gr_xlim)
#     gr_xticks=np.linspace(*gr_xlim,num=5)[:-1]
#     axs[7].set_xticks(gr_xticks)
#
#     return fig

if __name__ == '__main__':
#    data = [(1400,0.5,'MHz'),(2,0.1,'keV')]
    data = [(1400,0.5,'MHz'),
            ([2,10],[0.1,0.1],'keV'),
            ([3000,10000],[0.3,0.3],'angstrom')]
    r = sed(data)
    if r is not None:
        r.show()
