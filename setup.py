#!/usr/bin/env python

from distutils.core import setup


setup(name="sc_spectrum",
      version="0.0.0",
      description="Multilayer spectral clustering for CITEseq datasets",
      author="Jacob Leistico",
      url="",
      packages=["sc_spectrum", "sc_spectrum.cbmc", "sc_spectrum.bmnc"],
      )
